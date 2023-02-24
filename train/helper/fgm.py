import torch

class FGM():
    """
    定义对抗训练方法FGM,对模型embedding参数进行扰动
    """
    def __init__(self, model, epsilon=0.25,):
        # BERT模型
        self.model = model
        # 求干扰时的系数值
        self.epsilon = epsilon

        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        """
        得到对抗样本
        :param emb_name:模型中embedding的参数名
        :return:
        """
        # 循环遍历模型所有参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 把真实参数保存起来
                self.backup[name] = param.data.clone()
                # 对参数的梯度求范数
                norm = torch.norm(param.grad)
                # 如果范数不等于0并且norm中没有缺失值
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动，param.grad / norm=单位向量，起到了sgn(param.grad)一样的作用
                    r_at = self.epsilon * param.grad / norm
                    # 在原参数的基础上添加扰动
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        将模型原本的参数复原
        :param emb_name:模型中embedding的参数名
        """
        # 循环遍历模型所有参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 断言
                assert name in self.backup
                # 取出模型真实参数
                param.data = self.backup[name]
        # 清空self.backup
        self.backup = {}