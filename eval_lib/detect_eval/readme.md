# 检测算法评测脚本(P、R、Feasure)
##  评测
* 数据格式:

  gt: x1,y1,x2,y2,x3,y3,x4,y4; 
  pred: x1,y1,x2,y2,x3,y3,x4,y4,score
 
* 评测: python eval_open.py 
* iou阈值调整：修改concern/icdar2015_eval/detection/iou.py 中的iou_constraint