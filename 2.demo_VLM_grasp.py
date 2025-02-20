import cv2
import numpy as np
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
import base64
import json
import pyrealsense2 as rs
from ultralytics import YOLOWorld
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

# step1: 命令行获取用户输入的prompt：如：- 桌子上有哪些东西？- 把香蕉放进盒子里
# step2: 路由到对应函数获取输出
    # A、如果是桌子上有哪些东西？则只需要绘制显示，并打印出来
    # B、如果是把香蕉放进盒子里，则需要调用graspnet进行抓取

class VLM_GRASP:
    def __init__(self):
        # API_KEY
        self.API_KEY = "sk-761b43d7ea7342ba9336ec9e00b21051"
        # 当前使用的识别模型，如需要更换，可以在这里修改
        vision_model_list = ['qwen-vl-plus','yolo-world','grounding-dino-base']
        self.vision_model = vision_model_list[2]

        # initialize realsense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 1280x720,640x480,320x240
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        # Start streaming
        self.pipeline.start(self.config)

    def capture_image(self):
        '''
        获取realsense图像
        '''
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth.get_data())
        # resize to
        # color_image = cv2.resize(color_image, (500, 400))
        return color_image, depth_image

    def get_camera_paras(self):
        '''
        获取realsense相机参数
        '''
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intr_matrix = np.array([
            [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
        ]) 
        intr_coeffs = np.array(intr.coeffs)

        return intr_matrix, intr_coeffs

    def saveImg(self, color_image, depth_image):
        '''
        保存图像
        '''
        # save numpy
        # save the data
        np.save('./VLM_related/realsense_captured/color.npy', color_image)
        np.save('./VLM_related/realsense_captured/depth.npy', depth_image)
        # 保存图像
        colorFileName = './VLM_related/realsense_captured/color.jpg'
        depthFileName = './VLM_related/realsense_captured/depth.png'
        cv2.imwrite(colorFileName, color_image)
        cv2.imwrite(depthFileName, depth_image)
        return colorFileName, depthFileName

    def ask_LLM(self, content):
        '''
        调用大模型
        '''
        client = OpenAI(
            api_key=self.API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 向大模型发起请求
        completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": content
                }
            ]
            },
        ]
        )
        # 解析大模型返回结果
        result = completion.choices[0].message.content.strip()
        return result

    def route2func(self, prompt):
        '''
        根据用户输入的prompt，路由到对应的函数
        '''
        SYSTEM_PROMPT = '''
        我将给你一个字符串。你帮我输出函数名：
        1. 如果字符串是：桌子上有哪些东西？有什么？桌子上有什么等类似咨询问题？你就输出函数名：["list_objs"]
        2. 如果字符串是：把最大/黄色的/最小的/白色的物体放进盒子等类似抓取动作。你就输出函数名和描述物体的英文词：["grasp_obj","描述词"]，如：["grasp_obj","remote control"]
        注意：
        1. 只需要输出函数名本身，不需要任何其他数据

        我现在给你的字符串是：'''

        # 调用LLM
        func_name = self.ask_LLM(SYSTEM_PROMPT + prompt)
        # string转list
        func_name = eval(func_name)

        return func_name

    def ask_vlm(self, img_path, PROMPT):
        # 系统提示词
        SYSTEM_PROMPT = '''
        我将给你一张图片，以及一个指令：
        1. 如果指令是："list_objs"，你就：
        列出图中所有你能看到的物体，整理成JSON格式，格式如下：

        {
        "function":"list_objs",
        "objs":[
            ["类别名称",[左上角像素坐标x,左上角像素坐标y],[右下角像素坐标x,右下角像素坐标y]],
            ["类别名称",[左上角像素坐标x,左上角像素坐标y],[右下角像素坐标x,右下角像素坐标y]],
        ]
        }

        如：
        {
        "function":"list_objs",
        "objs":[
            ["apple",[100,100],[300,300]],
            ["banana",[120,100],[320,300]],
            ["apple",[120,120],[320,320]],
            ["banana",[420,400],[820,820]],
        ]
        }


        2. 如果指令是："grasp_obj"。你就输出：
        {
        "function":"mov_obj",
        "className":"类别名称",
        "xyxy":[[左上角像素坐标x,左上角像素坐标y],[右下角像素坐标x,右下角像素坐标y]],
        }
        如：指令是 把最大的橘子放进盒子里，你找到最大的橘子的坐标：
        {
        "function":"mov_obj",
        "className":"橘子",
        "xyxy":[[102,505],[324,860]],
        }

        3. 如果指令是："descibe_obj"。你就描述能看到的所有物体（不要忽略了一些小的物体），用逗号,分开，如
        {
        "function":"descibe_obj",
        "objs":["apple","banana","orange","milk box","mouse"]
        }


        注意：
        1. 只需要输出JSON本身，不需要任何其他数据，尤其是JSON前后的```符号
        2. 不要少了function

        现在指令是： '''

      
        client = OpenAI(
            api_key= self.API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 编码为base64数据
        with open(img_path, 'rb') as image_file:
            image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')
        # 向大模型发起请求``
        completion = client.chat.completions.create(
          model="qwen-vl-plus",
          # top_p=0.3,
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": SYSTEM_PROMPT  + PROMPT
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": image
                  }
                }
              ]
            },
          ]
        )
        
        # print(SYSTEM_PROMPT + RESOLUTION +"。\n指令是：" + PROMPT)
        # 解析大模型返回结果
        result = completion.choices[0].message.content.strip()
        
        return result

    def yoloWorld(self,img_path,classes):
       # Initialize a YOLO-World model
        model = YOLOWorld("./VLM_related/yolo-world/yolov8x-world.pt")  # or select yolov8m/l-world.pt for different sizes

        if classes:
            model.set_classes(classes)  # Set the classes for the model
        # model.set_classes(["apple", "orange", "banana", "pear"])  # Set the classes for the model
        # Execute inference with the YOLOv8s-world model on the specified image
        result =list( model.predict(img_path))[0]
        classes = result.names
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array

        dets = [] # 检测结果
        # 参考：https://docs.ultralytics.com/modes/predict/#boxes
        # 遍历每个框
        for box in boxes.data:
            l,t,r,b = map(float,box[:4]) # left, top, right, bottom
            conf, class_id =map(float, box[4:]) # confidence, class_id
            # 排除不需要追踪的类别
            dets.append({'bbox': [l,t,r,b], 'score': conf, 'class': classes[int(class_id)]})
            
        return dets
       

    def groundingDINO(self,img_path, classes):
        model_id = "./VLM_related/grounding-dino-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        image = Image.open(img_path)
        # Check for cats and remote controls
        # VERY important: text queries need to be lowercased + end with a dot
        text = ""
        for i in classes:
          text += i + "."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results

    def jsonOriganiner(self,json_str,PROMPT):
        '''
        将传过来的字符串整理成需要的json格式
        '''
        SYSTEM_PROMPT = '''
        我将给你一个字符串，和一个指令。
        如果指令是："list_objs"，你就将字符串整理成如下JSON格式：

        {
          "function":"list_objs",
          "objs":[
            ["className",[左上角像素坐标x,左上角像素坐标y],[右下角像素坐标x,右下角像素坐标y]],
            ["className",[左上角像素坐标x,左上角像素坐标y],[右下角像素坐标x,右下角像素坐标y]],
          ]
        }

        如：
        {
          "function":"list_objs",
          "objs":[
            ["apple",[100,100],[300,300]],
            ["banana",[120,100],[320,300]],
            ["apple",[120,120],[320,320]],
            ["banana",[420,400],[820,820]],
          ]
        }


        2. 如果指令是："mov_obj"，你就输出：
        {
          "function":"mov_obj",
          "className":"类别名称",
          "xyxy":[[左上角像素坐标x,左上角像素坐标y],[右下角像素坐标x,右下角像素坐标y]],
        }
        如：
        {
          "function":"mov_obj",
          "className":"橘子",
          "xyxy":[[102,505],[324,860]],
        }

        3. 如果指令是："descibe_obj"。你就输出：
        {
        "function":"descibe_obj",
        "objs":["类别名称","类别名称","类别名称","类别名称","类别名称"]
        }
        如：
        {
        "function":"descibe_obj",
        "objs":["apple","banana","orange","milk box","mouse"]
        }


        注意：
        1. 只需要输出JSON本身，不需要任何其他数据，尤其是JSON前后的```符号
        2. 类别要求是英文

        我现在给你的字符串是：'''
        # 调用LLM
        json_data = self.ask_LLM(SYSTEM_PROMPT + json_str + "。\n指令是：" + PROMPT)
        return json.loads(json_data)

    
    def saveDataAndCallGraspnet(self,data_list):
        '''
        这里保存数据，然后调用graspnet
        '''
        pass
      
# 测试
if __name__ == '__main__':
    demo =  VLM_GRASP()
    while True:
        # 获取用户输入的prompt
        prompt = input("请输入prompt：")
        # 获取图像
        print("正在获取图像...")
        color_image, depth_image = demo.capture_image()
        # 保存图像，并返回路径
        colorFilePath, depthFilePath = demo.saveImg(color_image, depth_image)
        # 判断当前使用的模型
        img = cv2.imread(colorFilePath)
        image_width, image_height = img.shape[:2]

        # 路由到对应的函数
        func_name = demo.route2func(prompt)
        # 执行不同的函数
        if func_name[0] == "list_objs":
            print("执行list_objs函数")
            

            if demo.vision_model == 'qwen-vl-plus':
                print("调用qwen-vl-plus，耗时约10s")
              # 调用VLM
                json_str = demo.ask_vlm(colorFilePath, "list_objs")
                # 整理成需要的json格式
                json_data = demo.jsonOriganiner(json_str, "list_objs")
                # 绘制            
                if json_data['function'] == 'list_objs':
                  for obj in json_data['objs']:
                      obj_type,lxy,rxy = obj[:3]
                      p1 = int(lxy[0] * image_width / 999),int( lxy[1] * image_height / 999)
                      p2 = int(rxy[0] * image_width / 999),int( rxy[1] * image_height / 999)
                      cv2.rectangle(img,p1,p2,(0,255,0),2)
                      cv2.putText(img,f"{obj_type}",p1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                # 先试用VLM获取有哪些物体
                json_str = demo.ask_vlm(colorFilePath, "descibe_obj")
                # 整理成需要的json格式
                json_data = demo.jsonOriganiner(json_str, "descibe_obj")
                class_names = json_data['objs']

                if demo.vision_model == 'yolo-world':
                    print("调用yolo-world，设置类别为：" + str(json_data['objs']))
                    results = demo.yoloWorld(colorFilePath, class_names)
                    for result in results:
                      bbox = result['bbox']
                      p1 = int(bbox[0]),int(bbox[1])
                      p2 = int(bbox[2]),int(bbox[3])
                      cv2.rectangle(img,p1,p2,(0,255,0),2)
                      cv2.putText(img,f"{result['class']}",p1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                elif demo.vision_model == 'grounding-dino-base':
                    
                    print("调用grounding-dino-base，设置类别为：" + str(json_data['objs']))
                    
                    results = demo.groundingDINO(colorFilePath, class_names)
                    for result in results:
                        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                            box = [int(i) for i in box]
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(img, f"{label} {score:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imwrite("result.jpg",img)
            
        elif func_name[0]  == "grasp_obj":
            class_name = func_name[1]
            print("调用grounding-dino-base，设置类别为：" + class_name)
            results = demo.groundingDINO(colorFilePath, [class_name])
            result_box = None
            for result in results:
                for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                    box = [int(i) for i in box]
                    result_box = box
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(img, f"{label} {score:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # 只需要1个物体，所以break
                    break
            cv2.imwrite("result.jpg",img)
            # 需要传递：彩图、深度图、检测框、相机参数
            # 获取相机参数
            intr_matrix, _ = demo.get_camera_paras()
            # 将intr_matrix和result_box保存到文件中
            np.save("./VLM_related/exchange/intr_matrix.npy", intr_matrix)
            np.save("./VLM_related/exchange/result_box.npy", result_box)
            # 通知graspnet进行抓取，通过文件是否存在来判断
            
            


