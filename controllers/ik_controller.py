from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from env.franka_env import FrankaEnv
from .base_controller import BaseController

MAX_STEPS = 2500

class IKController(BaseController):

    def __init__(self, env: FrankaEnv) -> None:
        super().__init__()
        self.env: FrankaEnv = env
        self.dt=5/2500
        self.stages=5
        self.state=0
        self.step_in_stage=0
        self.active_q=4
        self.active_q_index=[0,1,3,5]
        self.stages_steps=[100,250,250,250,250]
        self.static_steps=0

        self.init_q=self.env.arm_joint_positions
        self.resetting=False
        self.reset_q=[]
        self.fixed_q3=self.env.FIXED_JOINT_VALUES[0]
        self.fixed_q5=self.env.FIXED_JOINT_VALUES[1]
        self.fixed_q7=self.env.FIXED_JOINT_VALUES[2]

        self.num_q=7
        self.x=[0]*self.stages
        self.y=[0]*self.stages
        self.z=[0]*self.stages
        self.setState()
        self.selected_ik_solutions=np.zeros((self.stages, self.active_q))
        self.q_arr=[0]*(self.stages+1)
        self.q_arr[self.stages]=np.zeros((100, self.active_q))
        self.plan_all()


        

    def setState(self):
        Flange=0.107

        target_pos0=self.env.block_position.copy()
        target_pos0[2]+=self.env.block_size[2]*5
        self.x[0],self.y[0],self.z[0]=target_pos0

        target_pos1=self.env.block_position
        target_pos1[2]+=self.env.block_size[2]/2
        self.x[1],self.y[1],self.z[1]=target_pos1

        target_pos2=target_pos1.copy()
        target_pos2[2]+=self.env.table2_top_position[2]- self.env.tabletop_position[2] + 5*self.env.block_size[2]
        self.x[2],self.y[2],self.z[2]=target_pos2

        target_pos3=self.env.table2_top_position.copy()
        target_pos3[2]=target_pos2[2]
        self.x[3],self.y[3],self.z[3]=target_pos3

        target_pos4=self.env.target_position.copy()
        target_pos4[2]=self.env.target_position[2]+self.env.block_size[2]*1.3
        self.x[4],self.y[4],self.z[4]=target_pos4

        print("Begin task-----------")
        print("Block position: ", self.env.block_position)
        print("Table top position: ", self.env.tabletop_position)
        print("Table 2 top position: ", self.env.table2_top_position)
        print("Target positions: ",self.env.target_position)
        print("Calculated target positions: ")
        for state in range(self.stages):
            print(f"State {state}: ({self.x[state]}, {self.y[state]}, {self.z[state]})")

    def reset(self) -> None:
        super().reset()
        self.state=0
        self.step_in_stage=0
        self.env.set_arm_target(self.init_q)
        self.env.set_gripper(1)
        self.resetting=True
        self.plan_reset_route()
        self.env.reset()
        print("Resetting to initial position...")


    def is_done(self) -> bool:
        return self.state==self.stages 

    def compute_control(self) -> None:

        if self.resetting:
            if self.step_in_stage==self.reset_q.shape[0]:
                self.resetting=False
                self.step_in_stage=0
                print("Reset complete, resuming task.")
            else:
                self.env.set_arm_target(self.reset_q[self.step_in_stage])
                self.step_in_stage+=1
                return

        if self.step_in_stage == self.q_arr[self.state].shape[0]:
            self.state+=1
            self.step_in_stage=0
            print(f"  [{self.step_count:4d}] → Stage {self.state-1} reached")
            print(f"  Target position for stage {self.state-1}: ({self.x[self.state-1] if self.state-1<self.stages else 'N/A'}, {self.y[self.state-1] if self.state-1<self.stages else 'N/A'}, {self.z[self.state-1] if self.state-1<self.stages else 'N/A'})")
            print(f" Current finger position: {self.env.finger_joint_positions}")
            print(f"Current endeffector position: {self.env.endeffector_position}\n")
        
        if(self.state == self.stages):
            if(self.static_steps!=0):
                self.static_steps-=1
                return self.q_arr[self.stages-1][self.q_arr[self.stages-1].shape[0]-1]
            else: 
                return self.q_arr[self.stages-1][self.q_arr[self.stages-1].shape[0]-1]


        try:
            self.env.set_arm_target(self.q_arr[self.state][self.step_in_stage])
        except IndexError:
            print(f"Error: Index out of bounds for stage {self.state}")

        if(self.state==2 and self.step_in_stage==0):
            print("Gripper closed")
            print(f"Current finger position: {self.env.finger_joint_positions}")
            print(f"Current endeffector position: {self.env.endeffector_position}")
            self.env.set_gripper(0.2)


        if(self.state==self.stages-1 and self.step_in_stage== self.q_arr[self.state].shape[0]-1):
            self.env.set_gripper(1)
            
        self.step_in_stage+=1
        

    
    def plan(self, start_state, target_state):
        selected_ik_solutions1=self.selected_ik_solutions[start_state]
        target_pos=[self.x[target_state], self.y[target_state], self.z[target_state]]
        selected_ik_solutions2=self.compute_ik(target_pos,selected_ik_solutions1)
        self.selected_ik_solutions[target_state]=selected_ik_solutions2

        stage_steps=self.stages_steps[target_state]
        q=np.zeros((stage_steps, self.active_q))
        for i in range(self.active_q):
            q[:,i]=np.linspace(selected_ik_solutions1[i], selected_ik_solutions2[i], stage_steps)
        self.q_arr[target_state]=q

    def plan_origin(self, target_state):
        origin_q=np.array(self.env.arm_joint_positions,dtype=np.float64)
        selected_ik_solutions1=origin_q

        target_pos=[self.x[target_state], self.y[target_state], self.z[target_state]]
        selected_solution2=self.compute_ik(target_pos,origin_q)
        self.selected_ik_solutions[target_state]=selected_solution2
        
        stage_steps=self.stages_steps[target_state]
        q=np.zeros((stage_steps, self.active_q))
        for i in range(self.active_q):
            #q[:,i]=np.linspace(selected_ik_solutions1[i], selected_solution2[i], stage_steps)
            q[:,i]=self.LinearFunction_with_ParabolicBlends_JointSpace(selected_ik_solutions1[i], selected_solution2[i], stage_steps)
        self.q_arr[target_state]=q


    def LinearFunction_with_ParabolicBlends_JointSpace(self,origin,target,steps):
        s_acc=0.2
        s_const=1-2*s_acc

        if s_const>=0:
            r=s_acc/s_const
        else:
            r=0
        
        q_arr=np.zeros(steps)
 
        t_total=steps*self.dt
        t_const=  0 if s_const <= 0 else t_total/(4*r+1)
        t_acc= (t_total-t_const)/2

        a=2*s_acc/t_acc**2
        vs_max=a*t_acc

        if t_acc < 0:
            t_acc = t_total / 2
            t_const = 0

        already_time=0
        for j in range(steps):
            already_time=j*self.dt
            if(already_time < t_acc):
                s = 0.5 * a * already_time**2
            elif(already_time < t_acc + t_const):
                s = s_acc + vs_max * (already_time - t_acc)
            else:
                s = s_acc + s_const + vs_max * (already_time - t_acc - t_const) - 0.5 * a * (already_time - t_acc - t_const)**2
 
            s = min(s, 1.0)
            q_arr[j] = (1 - s) * origin + s * target

        return q_arr
        


    def plan_all(self):
        for target_state in range(self.stages):
            if target_state==0:
                self.plan_origin(target_state)
            else:
                self.plan(target_state-1, target_state)

    def plan_reset_route(self):
        steps=500
        origin_q=self.env.arm_joint_positions
        target_q=self.init_q
        q=np.zeros((steps, self.active_q))
        q=np.linspace(origin_q, target_q, steps)
        self.reset_q=q


    def dh_matrix(self, theta, d, a, alpha):
        """改进 DH 参数矩阵 (Modified DH)"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct,    -st,    0,     a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa,  ca,  d*ca],
            [0,     0,     0,    1]
        ])
    
    def forward_kinematics(self, q):
        dh_params = [
            (q[0], 0.333, 0, 0),
            (q[1], 0, 0, -np.pi / 2),
            (self.fixed_q3, 0.316, 0, np.pi / 2),
            (q[2], 0, 0.0825, np.pi / 2),
            (self.fixed_q5, 0.384, -0.0825, -np.pi / 2),
            (q[3], 0, 0, np.pi / 2),
            (self.fixed_q7, 0, 0.088, np.pi / 2),
            (0, 0.107, 0, 0),
            (0,0.1034,0,0),
        ]
        T = np.eye(4)
        for theta, d, a, alpha in dh_params:
            T = T @ self.dh_matrix(theta, d, a, alpha)
        return T[:3, 3]


    def compute_ik(self, target_pos,init_guess):
        q=np.array(init_guess,dtype=np.float64)
        eps=1e-6
        damping=1e-6
        max_iters=100
        tol=1e-9

        for _ in range(max_iters):
            pos=self.forward_kinematics(q)
            error=target_pos - pos
            if np.linalg.norm(error) < tol:
                return q

            J=np.zeros((3, self.active_q))
            for i in range(self.active_q):
                q_eps=np.copy(q)
                q_eps[i] += eps
                pos_eps=self.forward_kinematics(q_eps)
                J[:, i] = (pos_eps - pos) / eps
            
            J_pinv=J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
            q += J_pinv @ error

            q=np.arctan2(np.sin(q), np.cos(q))

        print("IK did not converge")
        print(f"Final error: {np.linalg.norm(error)}")
        print(f"Final joint angles: {q}")
        print(f"Target position: {target_pos}")
        return q

            