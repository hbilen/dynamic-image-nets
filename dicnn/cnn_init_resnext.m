% -------------------------------------------------------------------------
function net = cnn_init_resnext(net,opts)
% -------------------------------------------------------------------------
% initialize classifier
net = dagnn.DagNN.loadobj(net) ;

% convs = find(arrayfun(@(a) isa(a.block, 'dagnn.Conv'), net.layers)==1);

fclayer = net.getLayer('classifier_0') ;
sizeW = size(net.params(fclayer.paramIndexes(1)).value);

% opts.nCls = 101;
nCls = opts.nCls ;
DropOutRate = opts.DropOutRate ; 


net.params(fclayer.paramIndexes(1)).value = ...
  0.01 * randn([sizeW(1:3),nCls],'single') ;
net.params(fclayer.paramIndexes(2)).value = zeros(nCls,1,'single') ;


% change loss
softmax = find(arrayfun(@(a) isa(a.block, 'dagnn.SoftMax'), net.layers)==1);
if ~isempty(softmax)
  net.removeLayer(net.layers(softmax(1)).name) ;
end
% convs = find(arrayfun(@(a) isa(a.block, 'dagnn.Conv'), net.layers)==1);
fclayer = find(arrayfun(@(a) strcmp(a.name, 'classifier_0'), net.layers)==1);
net.renameVar(net.layers(fclayer(end)).name,'prediction') ;
net.renameVar('data','input') ;

%------------------------------------------------------------------------%
% configure appr-rank-pool
switch opts.pool1Type
  case 'arpool'
    if strcmp(opts.pool1Layer,'conv0')
      poolLyr1 = 1 ;
      net.addLayer('arpool',AppRankPooling('scale',0.1),{'input','VideoId1'},'DynImg');
      net.setLayerInputs(net.layers(1).name,{'DynImg'}) ;
    else
      poolLyr1 = find(arrayfun(@(a) strcmp(a.name, opts.pool1Layer), net.layers)==1);
      assert(~isempty(poolLyr1));
      net.addLayer('arpool',AppRankPooling('scale',0.1),{net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImg');
      net.setLayerInputs(opts.pool1Layer,{'DynImg'}) ;
    end
  case 'ppool1'
    if strcmp(opts.pool1Layer,'conv0')
      poolLyr1 = 1 ;
    else
      poolLyr1 = find(arrayfun(@(a) strcmp(a.name, opts.pool1Layer), net.layers)==1);
    end
    net.addLayer('parampool',LinComb('pad',[1 1 10 1]),...
      {'features_4_0_merge','VideoId1'},'DynImg0',{'conv0f','conv0b'});
    
%     net.params(end-1).value = 0.1 * ones(1,1,10,1,'single');
    net.params(end-1).value = 0.1 * randn(1,1,10,1,'single');
    net.params(end).value = zeros(1,1,'single');  
    
    net.addLayer('BnormDyn',dagnn.BatchNorm('numChannels',256),'DynImg0','DynImg',...
      {'dym','dyb','dybx'}) ;
    net.params(end-2).value =  ones(256,1,'single') ;
    net.params(end-1).value =  zeros(256,1,'single') ;
    net.params(end).value   =  zeros(256,2,'single') ;
    
%     net.addLayer('reluP',dagnn.ReLU(),...
%       {'DynImg1'},'DynImg');
    net.layers(16).inputs{1} = 'DynImg' ;
    for i=numel(net.params)-4:numel(net.params),
      net.params(i).learningRate = 0.1 * net.params(i).learningRate;
    end
  case 'none'
  otherwise
    error('Unknown pool type %s', opts.pool1Type) ;
end


net.rebuild() ;
%------------------------------------------------------------------------%
% second pool layer (max pooling)
% poolLyr2 = find(arrayfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
poolLyr2 = find(arrayfun(@(a) strcmp(a.name, 'features_7_1_merge'), net.layers)==1);
net.addLayer('tempPoolMax',TemporalPooling('method','max'),...
  {net.layers(poolLyr2(1)).outputs{1},'VideoId2'},'tempPoolMax');

% change the input of fc last layer
% net.setLayerInputs(net.layers(convs(end)).name,'tempPoolMax') ;
% net.addLayer('bnar',dagnn.BatchNorm('numChannels',2048),{'tempPoolMax'},...
%   'tempPoolMaxbn',{'bnar_m','bnar_b','bnar_x'});
poolLyr2next = find(arrayfun(@(a) strcmp(a.name, 'features_7_1_id_relu'), net.layers)==1);
net.setLayerInputs(net.layers(poolLyr2next(1)).name,{'tempPoolMax'}) ;
net.rebuild() ;
%------------------------------------------------------------------------%
% add drop-out layers
if DropOutRate>0

  pool5 = find(arrayfun(@(a) strcmp(a.name, 'features_8'), net.layers)==1);
  oo = net.layers(pool5(1)).outputs{1};
  net.addLayer('drop_pool5',dagnn.DropOut('rate',DropOutRate),...
    oo,sprintf('drop_%s',oo),{});
  net.setLayerInputs('classifier_permute',{sprintf('drop_%s',oo)}) ;
end


%------------------------------------------------------------------------%
% add multi-class error
net.addLayer('errMC',ErrorMultiClass(),{'prediction','label'},'mcerr');

net.addLayer('loss', ...
             LossNormalized('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

%------------------------------------------------------------------------%
net.rebuild()

% replace standard matconvnet bnorm with my version
bns = find(arrayfun(@(a) strcmp(class(a.block), 'dagnn.BatchNorm'), net.layers)==1);
for i=1:numel(bns)
  bb = net.layers(bns(i)).block ;
  net.layers(bns(i)).block = BatchNormN('numChannels',bb.numChannels,...
  'epsilon',bb.epsilon,...
  'opts',bb.opts) ;
end

% dagMergeBatchNorm(net) ;
% dagRemoveLayersOfType(net, 'dagnn.BatchNorm') ;
net_ = net.saveobj ;
net = dagnn.DagNN.loadobj(net_) ;
net.meta.normalization.border = [32 32] ;
