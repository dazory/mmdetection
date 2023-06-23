

class ParamManager():
    def __init__(self):
        super(ParamManager, self).__init__()

        self.hook_results = dict(module=dict(), input=dict(), output=dict())

    # Debuging tools
    def check_param_structure(self, model, blank=''):
        '''
        Example: check_param_structure(model)
        '''
        for name, child in model.named_children():
            print(f"{blank}{name}: ")
            self.check_param_structure(child, blank + '  ├ ')

    def check_grad_status(self, model, blank=''):
        '''
        Example: check_grad_status(model)
        '''
        for name, child in model.named_children():
            print(f"{blank}{name}: ", end='')
            for param in child.parameters():
                print(f"{param.requires_grad} ", end='')
            print('')
            self.check_grad_status(child, blank + '  ├ ')

    # Helper
    def find_layer(self, model, layer_name):
        '''
        Input:
                layer_name = (str)
        Example:
                layer_name = 'rpn.rpn_cls_layer'
                layer = find_layer(model, layer_name)
                print(layer)
        '''
        if isinstance(layer_name, str):
            layer_name = layer_name.split('.')

        if len(layer_name) != 0:
            for name, child in model.named_children():
                if name == layer_name[0]:
                    if len(layer_name) == 1:
                        return child
                    child = self.find_layer(child, layer_name[1:])
                    if child != False:
                        return child
        return False

    # Freeze
    def freeze_all_params(self, model):
        '''
        Example: freeze_all_params(model)
        '''
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_all_params(child)

    def melt_all_params(self, model):
        '''
        Example: melt_all_params(model)
        '''
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True
            self.melt_all_params(child)

    def freeze_params(self, model, layer_to_freeze_list):
        '''
        Output:
                freezed_layer_list = True if all layer in layer_to_freeze_list are freezed.
                                                         (list) else the list of freezed layers.
        Example:
                layer_to_freeze_list = ['rpn.rpn_cls_layer', 'rpn.rpn_reg_layer', 'rcnn_net.cls_layer', 'rcnn_net.reg_layer']
                print(freeze_params(model, layer_to_freeze_list))
        '''
        freezed_layer_list = []
        for layer_to_freeze in layer_to_freeze_list:
            layer = self.find_layer(model, layer_to_freeze)
            if layer != False:
                self.freeze_all_params(layer)
                freezed_layer_list.append(layer_to_freeze)

        return True if len(layer_to_freeze_list) == len(freezed_layer_list) \
            else freezed_layer_list

    def freeze_params_except(self, model, layer_not_to_freeze_list):
        '''
        Output:
            melted_layer_list = True if all layer in layer_not_to_freeze_list are melted and others are freezed.
                                (list) else the list of melted layers.
        Example:
            layer_not_to_freeze_list = ['rpn.rpn_cls_layer', 'rpn.rpn_reg_layer', 'rcnn_net.cls_layer', 'rcnn_net.reg_layer']
            print(freeze_params_except(model, layer_not_to_freeze_list))
        '''
        self.freeze_all_params(model)

        melted_layer_list = []
        for layer_to_melt in layer_not_to_freeze_list:
            layer = self.find_layer(model, layer_to_melt)
            if layer != False:
                self.melt_all_params(layer)
                melted_layer_list.append(layer_to_melt)

        return True if len(layer_not_to_freeze_list) == len(melted_layer_list) else melted_layer_list

    # Hook
    def register_forward_hook(self, model, layer_name_list, type='output', detach=False):
        def save_output(layer_name, type):
            if not type in self.hook_results.keys():
                raise TypeError(f"Only supported for 'module', 'input', and 'output as a type,"
                                f"but got {type}")
            def hook_fn(m, i, o):
                if type == 'module':
                    _data = m
                elif type == 'input':
                    _data = i
                elif type == 'output':
                    _data = o
                else:
                    raise TypeError('')
                if detach:
                    _data = _data.detach()
                self.hook_results[type].update({layer_name: _data})
            return hook_fn

        for layer_name in layer_name_list:
            layer = self.find_layer(model, layer_name)
            layer.register_forward_hook(save_output(layer_name, type))
