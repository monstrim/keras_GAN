class Adversarial(models.Model):
    '''
    This is a base class. It ~can~ be instantiated and setup by hand,
    but ideally you'll use a class extended from it which handles
    internally all the setup required for a specific framework 
    (such as GAN, CGAN, WGAN, BiGAN, etc).
    
    This class is idealized such that, when extending it, only
    __init__ needs definition in the child class;
    from there on, compiling and fitting should be the same for all
    children, and should work ~as closely as possible~ as the base
    keras model's equivalent methods to the end user.
    '''
    
    def __init__(self, *args, **kwargs):
        super(Adversarial, self).__init__(*args, **kwargs)
        self.players = {}
        self.metrics_names = []
        self.metrics_tensors = []
        self.metrics_updates = []
    
    
    def add_metric(self, name, tensor):
        self.metrics_names.append(name)
        self.metrics_tensors.append(tensor)
        
        
    def add_player(self, name, params, loss):
        assert name not in self.players, "Duplicate player: " + name
        self.players[name] = {
            'params': params,
            'loss':   loss
        }
    
    
    def compile(self,
                optimizer,
                **kwargs):
        
        if isinstance(optimizer, list):
            raise ValueError(
                'optimizer must be either a single value, '
                'or a dict with keys corresponding '
                'to players: \n(%s).\nFound: %s' 
                % (str(self.players.keys()), optimizer))
            
        elif not isinstance(optimizer, dict):
            optimizer = {key: optimizer for key in self.players.keys()}
            
        for name in optimizer:
            if name not in self.players:
                raise ValueError('Unknown player: %s' % name)
        for name in self.players:
            opt = optimizers.get(optimizer[name])
            self.players[name]['optimizer'] = opt
        self.loss = {}
        self.metrics = []
        self.loss_weights = None
        self.sample_weight_mode = None
        self.weighted_metrics = None
        self.target_tensors = None
        
        self._distribution_strategy = None
        self._grouped_model = None
        self._collected_trainable_weights = []
        
        if not self.built:
            # Model is not compilable because it does not know its number
            # of inputs and outputs, nor their shapes and names. 
            # We will compile after the first time the model 
            # gets called on training data.
            return
        self._is_compiled = True
        
        self.stateful_metric_names = []
        self.stateful_metric_functions = []
        
        self._function_kwargs = kwargs
        self._feed_targets = []
        self._feed_sample_weights = []
        self._feed_sample_weight_modes = []

        for name, player in self.players.items():
            self._collected_trainable_weights.extend(player['params'])
        
        self.loss_functions = [None for _ in self.outputs]
        
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss_fns = []
        
        self.loss_weights_list = [1. for _ in range(len(self.outputs))]
        
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        

        
    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError(
                'You must compile your model before using it.')
        if self.train_function is not None:
            return
        
        inputs = (self._feed_inputs + 
                  self._feed_targets + 
                  self._feed_sample_weights)
        if (self.uses_learning_phase 
                and not isinstance(K.learning_phase(), int)):
            inputs += [K.learning_phase()]
        
        with K.name_scope('training'):
            fns = []
            for name, player in self.players.items():
                opt = player['optimizer']
                params = player['params']
                loss = player['loss']
                
                with K.name_scope(name + '_' + opt.__class__.__name__):
                    updates = opt.get_updates(
                        params=params,
                        loss=loss
                    )
                fn = K.function(
                    inputs, [],
                    updates=updates,
                    name=name+'_train_function')
                fns.append(fn)
            
            output_fn = K.function(
                inputs, self.metrics_tensors, 
                name='train_function',
                **self._function_kwargs)
            
        def train_function(_inputs):
            for fn in fns:
                fn(_inputs)
            _out =  output_fn(_inputs)
            return _out
        
        self.train_function = train_function

