
# 创建socket服务器
class MotorControlServer:
    def __init__(self, resume=False, usb_id=0 ):
        self.motor_control = MotorControl(usb_id)  # 初始化电机控制实例

        if resume:
            # 恢复之前保存的电机角度
            last_degrees_raw = self.motor_control.read_motor_angles()
            self.motor_control.last_degrees = [last_degrees_raw[i] * self.motor_control.motor_ratios[i] for i in range(6)]
            print(f"恢复的电机角度：{last_degrees_raw}")

        else:
            # 先归位
            print('执行电机回零校准')
            result = self.motor_control.sequential_home_all()
            if result:
                print("所有电机回零成功！")
            else:
                print("回零过程中存在异常！请检查硬件")
                sys.exit(0)  # 退出程序
            # 回到垂直模式
            time.sleep(1)
            print('回到垂直模式')
            vertical_degrees = [180, 90, 83, 30, 110, 30]
            self.motor_control.execute_motors_degrees_normal(vertical_degrees)

        

        self.server_socket = None
        self.is_running = True  # 退出标志
    
    def handle_client(self, client_socket):
        """处理客户端发送的指令"""
        try:
            while self.is_running:
                # 接收数据长度
                data_length = client_socket.recv(8)
                if not data_length:
                    break
                data_length = int.from_bytes(data_length, byteorder='big')
                
                # 接收实际的数据
                data = b''
                while len(data) < data_length:
                    packet = client_socket.recv(data_length - len(data))
                    if not packet:
                        break
                    data += packet

                # 反序列化数据为Python对象
                command = pickle.loads(data)
                result = None
                try:
                    # 根据接收到的指令，调用不同的电机控制函数
                    if command['action'] == 'home':
                        print("执行归位操作...")
                        self.motor_control.move_home()
                        result = 5  # 假设归位操作需要5秒
                    elif command['action'] == 'gripper_on':
                        print("抓取...")
                        self.motor_control.gripper_on()
                        result = 0.05
                    elif command['action'] == 'gripper_off':
                        print("释放...")
                        self.motor_control.gripper_off()
                        result = 0.05
                    elif command['action'] == 'servo_gripper':
                        angle = command['params']
                        print(f"舵机夹爪，角度：{angle}")
                        self.motor_control.servo_gripper(angle)
                        result = 1
                    elif command['action'] == 'robodk_simu':
                        enable = True if command['params']== 1 else False
                        print(f"Robodk模拟：{enable}")
                        self.motor_control.robodk_simulation(enable)
                        result = 0.05
                    elif command['action'] == 'set_free_mode':
                        print(f"电机自由模式：{command['params']}")
                        result = self.motor_control.set_free_mode(command['params'])
                    elif command['action'] == 'get_motor_angles':
                        result = self.motor_control.read_motor_angles()
                        # print(f"获取电机编码器角度：{result}")
                    elif command['action'] == 'angle_mode':
                        # 移动一定的关节角度（相对于电机零位）
                        angles, speed_ratio = command['params']
                        print(f"角度模式，角度列表: {angles}, 速度比例：{speed_ratio}")
                        result = self.motor_control.execute_motors_degrees_normal(angles, speed_ratio)
                    

                    elif command['action'] == 'move_xyz_rotation':
                        # 移动到指定的XYZ、姿态（使用逆运动学）
                        xyz_rotation = command['params']
                        # 获取电机角度解
                        sol = self.motor_control.ik_normal_move(xyz_rotation[:3],xyz_rotation[3:],xyz_rotation[-1])
                        if sol is not None:
                            # 取2位小数
                            rounded_data = [round(x, 2) for x in sol]
                            print(f"普通运动模式，目标位置: {xyz_rotation[:3]}，目标姿态：{xyz_rotation[3:]}，电机角度解：{rounded_data}")
                            result = self.motor_control.execute_motors_degrees_normal(rounded_data)   
                        
                        else:
                            print(f"无解，目标位置: {xyz_rotation[:3]}，目标姿态：{xyz_rotation[3:]}")
                            result = -1

                    elif command['action'] == 'move_linear_xyz_rotation':
                        xyz_rotation = command['params']
                        # 计算电机移动角度列表
                        joint_positions_list = self.motor_control.ik_linear_move(xyz_rotation[:3],xyz_rotation[3:],xyz_rotation[-1])
                        time_list_group = []

                        if joint_positions_list is not None:
                            
                            t_total = 0
                            for degree_list in joint_positions_list:
                                rounded_data = [round(x, 2) for x in degree_list]
                                time_list = self.motor_control.calcualte_motors_time_for_one_step_linear(rounded_data)
                                # 提取总时间，累加
                                t_total += time_list[0]
                                time_list_group.append(time_list)
                            
                            print(f"直线模式总共需要时间: {t_total} 秒")
                            result = t_total
                            
                            
                        else:
                            print(f"无解，目标位置: {xyz_rotation[:3]}，目标姿态：{xyz_rotation[3:]}")
                            result = -1
                        # 先将计算结果发送回客户端，因为execute_motors_for_all_steps会阻塞
                        if result is not None:
                            response = pickle.dumps(result)
                            response_length = len(response)
                            client_socket.sendall(response_length.to_bytes(8, byteorder='big'))  # 发送数据长度
                            client_socket.sendall(response)  # 发送数据
                        # 执行运动
                        self.motor_control.execute_motors_for_all_steps(time_list_group)
                    
                    elif command['action'] == 'tiktok_rotate_point':
                        action_set = command['params']
                        # 计算电机移动角度列表
                        joint_positions_list = self.motor_control.tiktok_roate_around_point(action_set)
                        time_list_group = []

                        if joint_positions_list is not None:
                            
                            t_total = 0
                            for degree_list in joint_positions_list:
                                rounded_data = [round(x, 2) for x in degree_list]
                                time_list = self.motor_control.calcualte_motors_time_for_one_step_linear(rounded_data)
                                # 提取总时间，累加
                                t_total += time_list[0]
                                time_list_group.append(time_list)
                            
                            print(f"绕圈模式总共需要时间: {t_total} 秒")
                            result = t_total
                            
                            
                        else:
                            print(f"无解，目标位置: {xyz_rotation[:3]}，目标姿态：{xyz_rotation[3:]}")
                            result = -1
                        # 先将计算结果发送回客户端，因为execute_motors_for_all_steps会阻塞
                        if result is not None:
                            response = pickle.dumps(result)
                            response_length = len(response)
                            client_socket.sendall(response_length.to_bytes(8, byteorder='big'))  # 发送数据长度
                            client_socket.sendall(response)  # 发送数据
                        # 执行运动
                        self.motor_control.execute_motors_for_all_steps(time_list_group)

                    else:
                        print("未知命令")
                        result = None

                except Exception as e:
                    print(f"处理命令 {command['action']} 时发生错误: {e}")
                    raise  # 再次抛出以捕获更高层次的错误

                # 将计算结果发送回客户端
                # 普通模式不会阻塞，可以这里再发送
                if result is not None  and command['action'] !='move_linear_xyz_rotation' and command['action'] !='tiktok_rotate_point':
                    response = pickle.dumps(result)
                    response_length = len(response)
                    client_socket.sendall(response_length.to_bytes(8, byteorder='big'))  # 发送数据长度
                    client_socket.sendall(response)  # 发送数据

        except Exception as e:
            print(f"处理客户端指令时发生错误: {e}")
        finally:
            client_socket.close()

    def start_server(self, host='localhost', port=12345):
        """启动服务器，监听客户端连接"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1)  # 设置 1 秒的超时，用于非阻塞 accept
        print(f"服务器正在{host}:{port}监听...")

        while self.is_running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"接收到来自 {addr} 的连接")
                client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_handler.start()
            except socket.timeout:
                # 每次超时后检查是否需要退出
                continue

    def stop_server(self):
        """关闭服务器"""
        print("正在关闭服务器...")
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()



class EpisodeAPP:
    def __init__(self, ip='localhost', port=3001,):
      
        # 创建socket客户端
        self.server_address = (ip, port)
        # 1-6电机减速比分别是25、20、25、10、4、1
        self.motor_ratios = [25, 20, 25, 10, 4, 1]
        # 机械臂逆运动学
        degree_range_list = [
            [-180,140],
            [0,180],
            [-80,83],
            [-30,270],
            [-110,110],
            [-30,290],
        ]
        # Convert each element in the list to radians
        radian_range_list = [[np.deg2rad(degree) for degree in pair] for pair in degree_range_list]
        self.EPDobot = DHRobot(
            [
                            RevoluteDH(d=166,a= 55, alpha=np.pi/2,qlim=np.array(radian_range_list[0])),
                            RevoluteDH(d=0, a=200, alpha=0, qlim=np.array(radian_range_list[1])),
                            RevoluteDH(d=0, a=56, alpha=np.pi/2,qlim=np.array(radian_range_list[2])),
                            RevoluteDH(d=192, a=0, alpha=-np.pi/2,qlim=np.array(radian_range_list[3])),
                            RevoluteDH(d=0, a=0, alpha=np.pi/2,qlim=np.array(radian_range_list[4])),
                            RevoluteDH(d=55, a=0, alpha=0,qlim=np.array(radian_range_list[5])),
            ],
            name="EPDobot",
        )

     

    def send_command(self, command):
        """发送命令到服务器并接收返回值"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(self.server_address)

        # 序列化命令数据
        data = pickle.dumps(command)
        data_length = len(data)
        
        # 先发送数据长度（8字节）
        client_socket.sendall(data_length.to_bytes(8, byteorder='big'))
        # 发送实际的数据
        client_socket.sendall(data)

        # 接收服务器返回的数据长度
        response_length = client_socket.recv(8)
        if not response_length:
            client_socket.close()
            return None
        response_length = int.from_bytes(response_length, byteorder='big')

        # 接收实际的数据
        response_data = b''
        while len(response_data) < response_length:
            packet = client_socket.recv(response_length - len(response_data))
            if not packet:
                break
            response_data += packet

        # 反序列化服务器返回的结果
        result = pickle.loads(response_data)
        client_socket.close()
        return result
    def robodk_simulation(self,enable=1):
        """发送命令"""
        command = {'action': 'robodk_simu', 'params': enable}
        result = self.send_command(command)
        return result
    def set_free_mode(self,enable=1):
        """发送命令"""
        if enable == 1:
            print('机械臂10秒后进入自由移动模式，注意托举')
            time.sleep(10)
        command = {'action': 'set_free_mode', 'params': enable}
        result = self.send_command(command)
        
    def get_motor_degrees(self):
        """发送命令"""
        command = {'action': 'get_motor_angles', 'params': None}
        result = self.send_command(command)
        return result
    def gripper(self, on = 0):
        """发送抓取命令"""
        action_str = "gripper_on" if on > 0 else "gripper_off"
        command = {'action': action_str, 'params': None}
        result = self.send_command(command)
        time.sleep(0.05)
    def servo_gripper(self,degree):
        """发送舵机抓取命令"""
        action_str = "servo_gripper"
        command = {'action': action_str, 'params': degree}
        result = self.send_command(command)
        time.sleep(1)
        
    def move_robot_to_xyz_fixed_oritentaion(self, x, y, z):
        # 保证朝向
        """发送命令"""
        command = {'action': 'move_xyz_rotation', 'params': [x,y,z,90,0,180,'zyx']}
        result = self.send_command(command)
        # return result
        if result != -1:
            time.sleep(result)
            return 1
        else:
            return -1
        
    def move_robot_to_xyz_oritentaion(self, x, y, z, degrees_xyz_list=[], orientation='xyz'):
        """发送命令"""
        command = {'action': 'move_xyz_rotation', 'params': [x,y,z,degrees_xyz_list[0],degrees_xyz_list[1],degrees_xyz_list[2],orientation]}
        result = self.send_command(command)
        # return result
        if result != -1:
            time.sleep(result)
            return 1
        else:
            print("No IK solution")
            return -1
        
    def calculate_T_based_on_degrees(self,q_list):
        '''
        电机运动角度转为DH模型角度后求其次变换矩阵
        '''
        # 转为DH模型角度
        q1 = q_list[0]-180
        q2 = 180-q_list[1]
        q3 = 83-q_list[2]
        q4 = q_list[3]-30
        q5 = 110-q_list[4]
        q6 = q_list[5]-30
        
        # 正运动学
        return  self.EPDobot.fkine(np.deg2rad([q1,q2,q3,q4,q5,q6]))
    
    def angle_mode(self,angles, speed_ratio=1):
        """发送命令"""
        command = {'action': 'angle_mode', 'params': [angles,speed_ratio]}
        result = self.send_command(command)
        return result
    
    def tiktok_rotate_point(self,action_set=1):
        """发送命令"""
        command = {'action': 'tiktok_rotate_point', 'params':action_set}
        result = self.send_command(command)
        return result

    def convert_motor_angles(self,q_list):
        '''
        DH模型角度转为电机运动角度
        '''
        q1 = 180+q_list[0]
        q2 = 180-q_list[1]
        q3 = 83-q_list[2]
        q4 = 30+q_list[3]
        q5 = 110-q_list[4]
        q6 = 30+q_list[5]
        
        return q1,q2,q3,q4,q5,q6
    
    def check_xyz_euler(self,origin_T,xyz,euler):
        # 平移
        translation = SE3.Tx(xyz[0]) * SE3.Ty(xyz[1]) * SE3.Tz(xyz[2])

        # 旋转
        rotation = SE3.Rx(euler[0],unit='deg') * SE3.Ry(euler[1],unit='deg') * SE3.Rz(euler[2],unit='deg')
        # 综合平移和旋转
        pose = translation * rotation

        # 验证
        # print('原始T：',origin_T)
        # print('新T：',pose)
        if np.allclose(origin_T,pose):
            print('验证通过')

    def calculate_ee2base_T(self,xyz,rotation_zyx):
        translation = SE3.Tx(xyz[0]) * SE3.Ty(xyz[1]) * SE3.Tz(xyz[2])
        # 旋转, +/- 180效果一样，但是+- 90不一样
        rotation = SE3.Rz(rotation_zyx[0], unit='deg') * SE3.Ry(rotation_zyx[1], unit='deg') * SE3.Rx(rotation_zyx[2], unit='deg')
        # rotation1 = SE3.Rx(180, unit='deg') * SE3.Rz(-90, unit='deg')
        # 综合平移和旋转
        pose = translation * rotation
        return np.asarray(pose)
    

  