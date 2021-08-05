import torch
from model.Nets import Net
from model.build_model import Build_Model
"""
old_model = Build_Model(weight_path=None, resume=False, showatt=False, modify=False)
chkpt = torch.load("./nospp_last.pt", map_location="cpu")
# print(old_model)
pretrained_dict = chkpt["model"]

# print(pretrained_dict)
# 加载修改后的模型结构
new_model = Build_Model(modify=True)
model_dict = new_model.state_dict()

# 加载mobilenetv3 预训练模型

mv3_model = torch.load("./mobilenetv3.pth",map_location="cpu")
"""
def transmodel():
    old_model = Build_Model(weight_path=None, resume=False, showatt=False, modify=False)
    chkpt = torch.load("./last825.pt", map_location="cpu")
    # print(old_model)
    pretrained_dict = chkpt["model"]

    # print(pretrained_dict)
    # 加载修改后的模型结构
    new_model = Build_Model(modify=True)
    model_dict = new_model.state_dict()

    # 加载mobilenetv3 预训练模型

    mv3_model = torch.load("./mobilenetv3.pth", map_location="cpu")


    # 更新 预训练权重中 MobileNetV3 的权重值
    for k, v in mv3_model.items():
        for m, n in model_dict.items():
            # print(model_dict.keys().index(m))
            # print(m.split("submodule."))
            # if k == m[-len(k):] and isadd[i] == 0:
            if ("_submodule.features" in m) and k == m.split("submodule.")[1]:
                # if k == m.split("submodule.")[1]:
                    pretrained_dict[m] = v
                    break


    # 过滤不属于 model_dict的键
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    new_model.load_state_dict(model_dict)

    chkpt["model"] = model_dict

    print("////////////////////////////////////////")
    # print(chkpt)
    torch.save(chkpt, "last.pt")
    print("save success")

# new_model = Build_Model(modify=True)
# model_dict = new_model.state_dict()
#
# mv3_model = torch.load("./mobilenetv3.pth",map_location="cpu")
# for k,v in mv3_model.items():
#     print(k)
def test_trans():
    new_model = Build_Model(modify=True)
    # model_dict = new_model.state_dict()
    mlast = torch.load("./last.pt",map_location="cpu")
    new_model.load_state_dict(mlast["model"])
    print("success")

transmodel()
test_trans()