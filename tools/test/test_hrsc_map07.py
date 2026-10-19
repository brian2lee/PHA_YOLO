from ultralytics import YOLO
import numpy as np

# # 載入模型與執行驗證
model = YOLO('/ultralytics/runs/obb/          /weight/best.pt')
metrics = model.val(
    data='/ultralytics/ultralytics/cfg/datasets/hrsc.yaml',
    split="test",
    task="obb",
    conf=0.01,
    nms=True,
)

# VOC AP 計算函式
def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 從 metrics 擷取 PR 曲線
p_curve = metrics.box.p_curve  # shape: (nc, 100)
r_curve = metrics.box.r_curve  # shape: (nc, 100)

# 計算每個類別的 AP
ap_list = []
for class_id in range(p_curve.shape[0]):
    prec = p_curve[class_id]
    rec = r_curve[class_id]
    ap = voc_ap(rec, prec, use_07_metric=True)
    ap_list.append(ap)
    print(f"Class {class_id} VOC07 AP = {ap:.3f}")

# 計算 mean AP
map_07 = np.mean(ap_list)
print(f"\n===> mAP (VOC07 11-point) = {map_07:.3f}")
###########################################################################################################################################
# target_recalls = np.linspace(0.0, 1.0, 11)

# # confidence 門檻下的 p/r 曲線 (每一列對應一個類別)
# p_curve = metrics.box.p_curve  # shape: (nc, 100)
# r_curve = metrics.box.r_curve  # shape: (nc, 100)

# # 要插值的 recall 點
# target_recalls = np.linspace(0.0, 1.0, 11)
# ap_list = []

# for class_id in range(len(p_curve)):
#     r_vals = r_curve[class_id]
#     p_vals = p_curve[class_id]

#     # 排序 recall 用來做插值（防止 recall 非遞增）
#     sorted_indices = np.argsort(r_vals)
#     r_sorted = r_vals[sorted_indices]
#     p_sorted = p_vals[sorted_indices]

#     # 插值 precision 對應到指定 recall
#     interpolated_precisions = np.interp(target_recalls, r_sorted, p_sorted, left=0, right=0)
#     ap_11point = np.mean(interpolated_precisions)
#     ap_list.append(ap_11point)

#     print(f"\nClass {class_id}:")
#     for r, p in zip(target_recalls, interpolated_precisions):
#         print(f"Recall {r:.1f} -> Precision {p:.3f}")
#     print(f"AP (11-point interpolated) = {ap_11point:.3f}")

# # 計算所有類別的平均 AP
# map_11point = np.mean(ap_list)
# print(f"\n===> mAP (11-point interpolated over {len(ap_list)} classes): {map_11point:.3f}")
