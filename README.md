> 本项目是我为课程[《大模型视觉抓取6轴机械臂》](https://enpeicv.com/forum.php?mod=forumdisplay&fid=56) 设计的6轴机械臂。
>
> 主要目标是尽量用市面上的标准件制作一个性能优秀，价格优惠的教学用途机械臂（2000元以内）。
>
> If you have any questions, please submit issues or email me: enpeicv@outlook.com, have fun with it!
>
> 扫码加入微信WeChat交流群：
>
> <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202404161532376.png?x-oss-process=style/wp" style="width:200px;" />



![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502201233199.png?x-oss-process=style/wp)

## 一、Episode 1机械臂参数：

**名称：**Episode 1（意为第一代，可能根据学员需求，后续升级迭代）

**本体：**主要传动机构为铝合金和45号钢CNC加工零件，外壳为PLA 3D打印件。6自由度，6个42步进电机，4个高精度行星减速器，FOC闭环驱动板，CAN总线通信，负载500g，活动半径510mm，高重复精度。

**夹爪：**负压真空夹爪，25KG 舵机2指柔性夹爪。
**软件：**ROBODK上位机，正逆解算，拖拽示教，TCP 协议API编程，3D、6D抓取网络，多模态VLM、LLM等。
**SDK API：**TCP协议的API，支持各种编程语言，API有角度移动模式（MoveJ）、坐标位置模式、欧拉角模式、直线运动模式（MoveL）。
**应用：**已实现手眼标定（眼在手上、眼在手外）、3D、6D抓取等。

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111259136.png?x-oss-process=style/wp" style="zoom: 33%;" />

**视频演示**：[B站点击播放](https://www.bilibili.com/video/BV1KJwNepESk/?spm_id_from=333.1387.0.0&vd_source=39b1662212679b11469d17d3bee8df4e)

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502201231788.png?x-oss-process=style/wp" style="zoom: 25%;" />

## 二、原理图：

**整体：**

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502201141412.png?x-oss-process=style/wp)

**电机驱动：**

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502141158971.png?x-oss-process=style/wp" style="zoom:50%;" />

**夹爪：**



## 三、物料表

| 零件编号 | 图示                                                         | 名称             | 类型     | 1成品实际数量 | 作用                                 |
| -------- | ------------------------------------------------------------ | ---------------- | -------- | ------------- | ------------------------------------ |
| 10003    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111626570.png?x-oss-process=style/wp" style="zoom:33%;" /> | 小轴承支撑       | 3D打印   | 1             | 支撑两个轴承                         |
| 10009    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111627635.png?x-oss-process=style/wp" style="zoom:33%;" /> | J2腰关节         | 3D打印   | 1             | 腰关节                               |
| 10010    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111630060.png?x-oss-process=style/wp" style="zoom:33%;" /> | J6腕部           | 3D打印   | 1             | 腕部                                 |
| 10013    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111633629.png?x-oss-process=style/wp" style="zoom:33%;" /> | 磁铁组装辅助     | 3D打印   | 1             | 用来安装电机后的磁铁                 |
| 10016    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111633673.png?x-oss-process=style/wp" style="zoom:33%;" /> | 法兰组装辅助     | 3D打印   | 1             | 用于辅助安装法兰螺丝                 |
| 10017    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111634111.png?x-oss-process=style/wp" style="zoom:33%;" /> | J1底座闸门       | 3D打印   | 1             | 底座闸门，用于固定各种插座           |
| 10019    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111634980.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5盖子皮带面     | 3D打印   | 1             | 盖子                                 |
| 10020    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111634051.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5盖子接线面     | 3D打印   | 1             | 盖子                                 |
| 10021    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111637749.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5小臂旋转       | 3D打印   | 1             | 小臂旋转                             |
| 10024    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111637093.png?x-oss-process=style/wp" style="zoom:33%;" /> | J4盖子           | 3D打印   | 1             | 小臂固定盖子                         |
| 10025    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111637514.png?x-oss-process=style/wp" style="zoom:33%;" /> | J4小臂固定       | 3D打印   | 1             | 小臂固定                             |
| 10028    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111638353.png?x-oss-process=style/wp" style="zoom:33%;" /> | J3大臂           | 3D打印   | 1             | 大臂                                 |
| 10031    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111638566.png?x-oss-process=style/wp" style="zoom:33%;" /> | J1固定底座       | 3D打印   | 1             | 固定底座                             |
| 10033    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111642863.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5 60齿同步轮    | 3D打印   | 1             | 同步带轮                             |
| 10043    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111642926.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5轴承盖子       | 3D打印   | 1             | 轴承盖子                             |
| 10044    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111642661.png?x-oss-process=style/wp" style="zoom:33%;" /> | J6端侧安装面     | 3D打印   | 1             | 机械臂末端，用来安装各种工具、执行器 |
| 10045    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111642666.png?x-oss-process=style/wp" style="zoom:33%;" /> | J6盖子           | 3D打印   | 1             | 腕部盖子                             |
| 10046    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111642559.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5电机拉紧板     | 3D打印   | 1             | 用来拉紧同步带                       |
| 10002    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111642607.png?x-oss-process=style/wp" style="zoom:33%;" /> | 键槽法兰         | CNC加工  | 3             | 用于传递扭矩                         |
| 10007    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111645363.png?x-oss-process=style/wp" style="zoom:33%;" /> | 腰关节铝片正面   | CNC加工  | 1             | 用来固定电机                         |
| 10008    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111645203.png?x-oss-process=style/wp" style="zoom:33%;" /> | 腰关节铝片背面   | CNC加工  | 1             | 用来固定电机                         |
| 10022    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111645506.png?x-oss-process=style/wp" style="zoom:33%;" /> | 小臂铝片正面     | CNC加工  | 1             | 用来固定电机                         |
| 10023    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111645649.png?x-oss-process=style/wp" style="zoom:33%;" /> | 小臂铝片背面     | CNC加工  | 1             | 用来固定电机                         |
| 10027    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111645259.png?x-oss-process=style/wp" style="zoom:33%;" /> | 大臂铝板         | CNC加工  | 1             | 支撑大臂，连接电机、法兰             |
| 10001    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111713604.png?x-oss-process=style/wp" style="zoom: 25%;" /> | 电机驱动         | 采购     | 6             | 电机驱动，这里用的是张大头家CAN套餐  |
| 10004    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111751746.png?x-oss-process=style/wp" style="zoom:33%;" /> | 小轴承           | 采购     | 4             | 轴承                                 |
| 10005    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111716589.png?x-oss-process=style/wp" style="zoom: 25%;" /> | 自攻螺钉M2×8     | 采购     | 20            | 用于紧固盖子                         |
| 10006    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111717779.png?x-oss-process=style/wp" style="zoom:25%;" /> | 自攻螺丝M6×16    | 采购     | 4             | 用于紧固固定底座，比如固定在木板上   |
| 10011    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111715391.png?x-oss-process=style/wp" style="zoom: 15%;" /> | 胶水             | 采购     | 1             | 用于将磁铁站在电机轴上               |
| 10012    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111718468.png?x-oss-process=style/wp" style="zoom:33%;" /> | 联轴器           | 采购     | 1             | 联轴器                               |
| 10014    | /                                                            | 电子线-绿蓝      | 采购     | 0.35m         | CAN信号线                            |
| 10015    | /                                                            | 电子线-红黑      | 采购     | 0.35m         | 电源正负线                           |
| 10018    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111719616.png?x-oss-process=style/wp" style="zoom:33%;" /> | 带灯按钮开关     | 采购     | 1             | 开关                                 |
| 10026    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111721509.png?x-oss-process=style/wp" style="zoom:33%;" /> | 倒边螺母M3x8     | 采购     | 12            | 连接腕部与轴承                       |
| 10029    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111731245.png?x-oss-process=style/wp" style="zoom:33%;" /> | 导向轴支座       | 采购     | 1             | 连接末端与电机输出轴                 |
| 10030    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111732724.png?x-oss-process=style/wp" style="zoom:33%;" /> | 固定环           | 采购     | 1             | 连接小臂旋转与电机输出轴             |
| 10032    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111732584.png?x-oss-process=style/wp" style="zoom:33%;" /> | 回转支承         | 采购     | 1             | 固定底座与腰部连接部件               |
| 10034    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111731503.png?x-oss-process=style/wp" style="zoom:33%;" /> | 同步带           | 采购     | 1             | 同步带                               |
| 10035    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111733296.png?x-oss-process=style/wp" style="zoom:33%;" /> | 光轴6x30         | 采购     | 1             | 配合联轴器，扩展轴长度               |
| 10036    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111733853.png?x-oss-process=style/wp" style="zoom:33%;" /> | USB2CAN转换器    | 采购     | 1             | 插在电脑控制机械臂                   |
| 10038    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111735110.png?x-oss-process=style/wp" style="zoom: 15%;" /> | 电机L60/20:1     | 采购     | 1             | 第2个关节电机、减速器                |
| 10039    | 类似10038                                                    | 电机L48/25:1     | 采购     | 1             | 第1个关节电机、减速器                |
| 10040    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502141248325.png?x-oss-process=style/resize" style="zoom: 33%;" /> | 电机L40          | 采购     | 2             | 第5、6个关节电机                     |
| 10041    | 类似10038                                                    | 电机L34/25:1     | 采购     | 1             | 第3个关节电机、减速器                |
| 10042    | 类似10038                                                    | 电机L34/10:1     | 采购     | 1             | 第4个关节电机、减速器                |
| 10047    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111736145.png?x-oss-process=style/wp" style="zoom:33%;" /> | J5 15齿同步轮    | 采购     | 1             | 同步带轮，连接电机与同步带           |
| 10048    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111736025.png?x-oss-process=style/wp" style="zoom:25%;" /> | 紧定螺钉M6×10    | 采购     | 4             | 紧定键槽法兰与电机输出轴             |
| 10049    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111737219.png?x-oss-process=style/wp" style="zoom:25%;" /> | 螺钉M8×25        | 采购     | 15            | 螺钉                                 |
| 10050    | /                                                            | 螺钉M6×16        | 采购     | 4             | 螺钉                                 |
| 10051    | /                                                            | 螺钉M5×50        | 采购     | 2             | 螺钉                                 |
| 10052    | /                                                            | 螺钉M5×25        | 采购     | 2             | 螺钉                                 |
| 10053    | /                                                            | 螺钉M5×16        | 采购     | 9             | 螺钉                                 |
| 10054    | /                                                            | 螺钉M4×10        | 采购     | 13            | 螺钉                                 |
| 10055    | /                                                            | 螺钉M3×8         | 采购     | 6             | 螺钉                                 |
| 10056    | /                                                            | 螺钉M3×60        | 采购     | 8             | 螺钉                                 |
| 10057    | /                                                            | 螺钉M3×5         | 采购     | 6             | 螺钉                                 |
| 10058    | /                                                            | 螺钉M3×40        | 采购     | 20            | 螺钉                                 |
| 10059    | /                                                            | 螺钉M3×35        | 采购     | 8             | 螺钉                                 |
| 10060    | /                                                            | 螺钉M3×20        | 采购     | 2             | 螺钉                                 |
| 10061    | /                                                            | 螺钉M3×16        | 采购     | 8             | 螺钉                                 |
| 10062    | /                                                            | 螺钉M3×10        | 采购     | 12            | 螺钉                                 |
| 10064    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111738117.png?x-oss-process=style/wp" style="zoom:25%;" /> | 螺母M5           | 采购     | 14            | 螺母                                 |
| 10065    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111738856.png?x-oss-process=style/wp" style="zoom:25%;" /> | 螺母M3           | 采购     | 19            | 螺母                                 |
| 10066    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111739626.png?x-oss-process=style/wp" style="zoom:25%;" /> | DC 12V 插座      | 采购     | 1             | 12V适配器电源插座                    |
| 10067    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111751961.png?x-oss-process=style/wp" style="zoom:25%;" /> | 4P排插           | 采购     | 1             | 为夹爪控制盒预留信号线和电源线       |
| 10068    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111739308.png?x-oss-process=style/wp" style="zoom:15%;" /> | 3x3x14键         | 采购     | 4             | 插入键槽法兰                         |
| 10069    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111740591.png?x-oss-process=style/wp" style="zoom:15%;" /> | 3x7x6 垫片       | 采购     | 6             | J1电机螺母与驱动之间                 |
| 10070    | /                                                            | 3x7x2 垫片       | 采购     | 40            | 垫片                                 |
| 10071    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111741088.png?x-oss-process=style/wp" style="zoom:25%;" /> | 2芯软线          | 采购     | 1.5m          | 连接USB2CAN转接头与2P排插            |
| 10072    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111741390.png?x-oss-process=style/wp" style="zoom:25%;" /> | 2P排插           | 采购     | 1             | 连接2芯软线                          |
| 10073    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111742192.png?x-oss-process=style/wp" style="zoom:25%;" /> | 12V电源          | 采购     | 1             | 12V电源适配器                        |
| 10074    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111743050.png?x-oss-process=style/wp" style="zoom:25%;" /> | 3MM扳手          | 采购     | 1             | 紧固 键槽法兰的紧定螺钉              |
| 10075    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111744758.png?x-oss-process=style/wp" style="zoom:25%;" /> | 2.5MM扳手        | 采购     | 1             | 紧固小臂旋转                         |
| 10076    | /                                                            | 1.5MM扳手        | 采购     | 1             | 紧固末端导向轴支座                   |
| 10077    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111745667.png?x-oss-process=style/wp" style="zoom:25%;" /> | 磁铁             | 采购     | 6             | 安装在电机末端，用于驱动器测量角度   |
| 10078    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502111746082.png?x-oss-process=style/wp" style="zoom:25%;" /> | 镊子             | 采购     | 1             | 用于夹取部分零件                     |
| 10079    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502151814983.png?x-oss-process=style/wp" style="zoom: 25%;" /> | 驱动-8P通讯线    | 驱动配套 | 6             | 驱动配套                             |
| 10080    | 同上                                                         | 驱动-4P电机线    | 驱动配套 | 6             | 驱动配套                             |
| 10081    | 同上                                                         | 驱动-注塑外壳    | 驱动配套 | 6             | 驱动配套                             |
| 10082    | 同上                                                         | 驱动-CAN模块     | 驱动配套 | 6             | 驱动配套                             |
| 10083    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502151817124.png?x-oss-process=style/resize" style="zoom:25%;" /> | 热缩管           | 采购     | 0.6m          | 接电子线用                           |
| 10084    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502151816471.png?x-oss-process=style/resize" style="zoom: 25%;" /> | 包线管           | 采购     | 0.6m          | 包裹电子线                           |
| 10085    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502151822805.png?x-oss-process=style/resize" style="zoom: 25%;" /> | 4P连接线对插     | 采购     | 1             | 接电子线用                           |
| 10086    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502151820067.png?x-oss-process=style/resize" style="zoom:25%;" /> | 长款电机配套4P线 | 采购     | 1             | 接电子线用                           |
| 10087    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502151823957.png?x-oss-process=style/resize" style="zoom:25%;" /> | 快速接线端子     | 采购     | 4             | 接电子线用                           |
| 10088    | <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202502191650155.png?x-oss-process=style/wp" style="zoom:15%;" /> | 3*7*1金属垫片    | 采购     | 6             | 垫片                                 |



## 四、API简介

**链接机械臂**：`0.server.py`

示例代码：

```python
# 导入控制API
from episode_controller.episodeServer import MotorControlServer
......
# 配置服务器
parser = argparse.ArgumentParser(description="Motor Control Server")
# resume：是否在上次姿态恢复
parser.add_argument("--resume", type=bool, default=False, help="Resume previous session")
# usb_id：USB地址，最高支持6个机械臂同时驱动
parser.add_argument("--usb_id", type=int, default=1, help="USB handler ID, use pcaninfo to check")
# TCP协议IP地址
parser.add_argument("--ip", type=str, default="localhost", help="Server IP address")
# 端口
parser.add_argument("--port", type=int, default=12345, help="Server port")

# 启动TCP协议API服务器
server = MotorControlServer(resume=args.resume, usb_id=args.usb_id)
server.start_server(host=args.ip, port=args.port)
```

这样启动后，便可以使用支持TCP协议的编程语言去连接操作机械臂。



**API列表**：`episodeServer.py`

* home: 执行电机回零校准，将机械臂移动到初始位置
* gripper: 控制夹爪动作，根据参数决定抓取（闭合）或释放（打开）
* servo_gripper: 发送舵机抓取命令，设置夹爪角度
* robodk_simulation: 启动或关闭Robodk模拟功能
* set_free_mode: 设置电机自由模式，允许手动或自由控制电机
* get_motor_angles: 获取当前各电机的角度信息
* angle_mode: 通过角度模式直接控制各电机运动
* move_xyz_rotation: 使用逆运动学移动到指定XYZ位置
* move_linear_xyz_rotation: 沿直线路径移动到目标位置

```python
def home(self):
    """
    执行电机回零校准，将机械臂移动到初始位置。

    返回:
    - 归位操作所需的时间（秒）。
    """
    command = {'action': 'home', 'params': None}
    result = self.send_command(command)
    if result is not None:
        print("正在归位，预计耗时 {} 秒".format(result))
        time.sleep(result)
    return result

def gripper(self, on=0):
    """
    控制夹爪的抓取和释放。

    参数:
    - on: 大于0表示抓取（闭合），否则表示释放（打开）。

    返回:
    - 机械臂响应的结果。
    """
    action_str = "gripper_on" if on > 0 else "gripper_off"
    command = {'action': action_str, 'params': None}
    result = self.send_command(command)
    time.sleep(0.05)
    return result

def servo_gripper(self, degree):
    """
    控制舵机夹爪动作，设置指定角度。

    参数:
    - degree: 舵机的目标角度（单位：度）。

    返回:
    - 机械臂响应的结果。
    """
    command = {'action': 'servo_gripper', 'params': degree}
    result = self.send_command(command)
    time.sleep(1)  # 等待动作完成
    return result

def robodk_simulation(self, enable):
    """
    启动或关闭Robodk模拟功能。

    参数:
    - enable: 布尔值，True表示启用模拟，False表示禁用。

    返回:
    - 机械臂响应的结果。
    """
    params = 1 if enable else 0
    command = {'action': 'robodk_simu', 'params': params}
    result = self.send_command(command)
    time.sleep(0.05)
    return result

def set_free_mode(self, mode):
    """
    设置电机自由模式，允许手动或自由控制电机，用来示教模式。

    参数:
    - mode: 自由模式参数，根据具体需求设定。

    返回:
    - 机械臂响应的结果。
    """
    command = {'action': 'set_free_mode', 'params': mode}
    result = self.send_command(command)
    return result

def get_motor_angles(self):
    """
    获取当前各电机编码器的角度值。

    返回:
    - 电机角度列表，如果获取失败则返回None。
    """
    command = {'action': 'get_motor_angles', 'params': None}
    result = self.send_command(command)
    return result

def angle_mode(self, angles, speed_ratio=1.0):
    """
    角度模式运动：根据指定角度直接控制各电机的运动。

    参数:
    - angles: 电机角度列表，顺序与实际电机对应。
    - speed_ratio: 运动速度比例，默认1.0。

    返回:
    - 机械臂响应的结果。
    """
    command = {'action': 'angle_mode', 'params': (angles, speed_ratio)}
    result = self.send_command(command)
    return result

def move_xyz_rotation(self, x, y, z, rotation_angles=[90, 0, 180], order='zyx'):
    """
    使用逆运动学控制机械臂运动到指定的XYZ位置，并设置目标姿态。

    参数:
    - x, y, z: 目标位置坐标（单位：毫米或其他，根据实际设定）。
    - rotation_angles: 目标姿态角度列表（默认为 [90, 0, 180]）。
    - order: 旋转顺序（如'zyx'），确保与控制端一致。

    返回:
    - 如果有解，返回运动执行时间（秒），否则返回 -1 表示无解。
    """
    params = [x, y, z] + rotation_angles + [order]
    command = {'action': 'move_xyz_rotation', 'params': params}
    result = self.send_command(command)
    if result != -1 and result is not None:
        print("运动中，预计耗时 {} 秒".format(result))
        time.sleep(result)
    else:
        print("目标位置或姿态无解")
    return result

def move_linear_xyz_rotation(self, x, y, z, rotation_angles=[90, 0, 180], order='zyx'):
    """
    直线模式运动：沿直线路径运动到指定位置，并同时调整末端姿态。

    参数:
    - x, y, z: 目标位置坐标。
    - rotation_angles: 目标姿态角度列表。
    - order: 旋转顺序（默认为 'zyx'）。

    返回:
    - 如果运动规划成功，返回运动总时间（秒）；否则返回 -1 表示无解。
    """
    params = [x, y, z] + rotation_angles + [order]
    command = {'action': 'move_linear_xyz_rotation', 'params': params}
    result = self.send_command(command)
    if result != -1 and result is not None:
        print("直线运动规划成功，总共需要 {} 秒".format(result))
        time.sleep(result)
    else:
        print("直线运动规划失败：目标无解")
    return result
```



## 五、应用代码

### 5.1 示教模式

> 操作其中一台机械臂，另外一台跟着运动。

代码：`3.teach_mode.py`

### 5.2 3D抓取

> 眼在手外深度相机辅助3D平面抓取（末端欧拉角固定）

代码：`2.air_sucker.py`

### 5.3 6D抓取

> 眼在手上深度相机辅助3D平面抓取（末端欧拉角不固定）

代码：`1.demo_episode_with_gripper.py`

### 5.3 结合VLM、LLM

> 结合大语言模型、多模态模型

代码：

* `2.demo_VLM_grasp.py`
* `3.demo_VLM_handler.py`