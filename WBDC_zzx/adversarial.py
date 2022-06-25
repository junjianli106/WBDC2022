import torch

class FGM():
    def __init__(self, args, model):
        self.model = model
        self.args = args
        self.backup = {}

    def attack(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # bert.embeddings.word_embeddings.weight
        # bert.embeddings.position_embeddings.weight
        # bert.embeddings.token_type_embeddings.weight
        # bert.embeddings.LayerNorm.weight
        # bert.embeddings.LayerNorm.bias
        text_epsilon = self.args.text_fgm_eps
        img_epsilon = self.args.img_fgm_eps
        for name, param in self.model.named_parameters():
            if name.startswith('swa_model.'):
                continue
            if param.requires_grad and (emb_name in name or 'visual' in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    if 'visual' in name:
                        r_at = img_epsilon * param.grad / norm
                        param.data.add_(r_at)
                    else:
                        r_at = text_epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if name.startswith('swa_model.'):
                continue
            if param.requires_grad and (emb_name in name or 'visual' in name):
                assert name in self.backup
                
                param.data = self.backup[name]
        self.backup = {}
        
        
class PGD:
    def __init__(self, args, model):
        self.model = model
        self.args = args
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embeddings.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        text_alpha = self.args.text_pgd_alpha
        img_alpha = self.args.img_pgd_alpha
        epsilon = self.args.pgd_epsilon
        for name, param in self.model.named_parameters():
            if name.startswith('swa_model.'):
                continue
            if param.requires_grad and (emb_name in name or 'visual' in name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    alpha = text_alpha if emb_name in name else img_alpha
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if name.startswith('swa_model.'):
                continue
            if param.requires_grad and (emb_name in name or 'visual' in name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):  # bound constraint
        r = param_data - self.emb_backup[param_name]
        
        if torch.norm(r) > epsilon:
            # print(torch.norm(r))
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]