% -------------------------------------------------------------------------
function net = cnn_init_cafferef(net,opts)
% -------------------------------------------------------------------------

drop6p = find(cellfun(@(a) strcmp(a.name, 'dropout6'), net.layers)==1);
drop7p = find(cellfun(@(a) strcmp(a.name, 'dropout7'), net.layers)==1);

if ~isempty(drop6p)
  assert(~isempty(drop7p));
  net.layers{drop6p}.rate = opts.DropOutRate;
  net.layers{drop7p}.rate = opts.DropOutRate;
else
  relu6p = find(cellfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
  relu7p = find(cellfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);

  drop6 = struct('type','dropout','rate', opts.DropOutRate,'name','dropout6') ;
  drop7 = struct('type','dropout','rate', opts.DropOutRate,'name','dropout7') ;
  net.layers = [net.layers(1:relu6p) drop6 net.layers(relu6p+1:relu7p) drop7 net.layers(relu7p+1:end)];
end

% replace fc8
fc8l = cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1;

nCls = opts.nCls ;
% nCls = 101;
sizeW = size(net.layers{fc8l}.weights{1});

if sizeW(4)~=nCls
  net.layers{fc8l}.weights = {zeros(sizeW(1),sizeW(2),sizeW(3),nCls,'single'), ...
    zeros(1, nCls, 'single')};
end

% change loss
% net.layers(end) = [];
net.layers{end} = struct('name','loss', 'type','softmaxloss') ;

% convert to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

poolLyr1 = find(arrayfun(@(a) strcmp(a.name, opts.pool1Layer), net.layers)==1);
assert(~isempty(poolLyr1));
% configure appr-rank-pool
switch opts.pool1Type
  case 'arpool'
    if strcmp(opts.pool1Layer,'conv1')
      net.addLayer('arpool',AppRankPooling('scale',1),{net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImgN');
      net.addLayer('l2normalize',L2Normalize('scale',6000,'clip',[-128 128]),...
        'DynImgN','DynImg');
    else
      net.addLayer('arpool',AppRankPooling('scale',0.1),{net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImgN');
      net.addLayer('reluP',dagnn.ReLU(),...
      {'DynImgN'},'DynImg');
    end
    net.setLayerInputs(opts.pool1Layer,{'DynImg'}) ;  
  case 'ppool1'
    if strcmp(opts.pool1Layer,'conv1')
      net.addLayer('parampool',LinComb('pad',[1 1 10 1]),...
      {net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImg',{'conv0f','conv0b'});
    else
      net.addLayer('parampool',LinComb('pad',[1 1 10 1]),...
      {net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImgN',{'conv0f','conv0b'});
    net.addLayer('reluP',dagnn.ReLU(),...
      {'DynImgN'},'DynImg');
    end
    
    net.layers(poolLyr1).inputs{1} = 'DynImg' ;
%     net.params(end-1).value = 0.01 * randn(1,1,10,1,'single');
    net.params(end-1).value = 0.1 * ones(1,1,10,1,'single');
    net.params(end).value = zeros(1,1,'single');    
    
    net.params(end-1).learningRate = 0.1 ;
    net.params(end).learningRate = 0.2 ;
  case 'ppool2'
    if strcmp(opts.pool1Layer,'conv1')
      net.addLayer('parampool',LinComb('pad',[1 1 10 1]),...
      {net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImg',{'conv0f','conv0b'});
    else
      net.addLayer('parampool',LinComb('pad',[1 1 10 1]),...
      {net.layers(poolLyr1).inputs{1},'VideoId1'},'DynImgN',{'conv0f','conv0b'});
    net.addLayer('reluP',dagnn.ReLU(),...
      {'DynImgN'},'DynImg');
    end
    
    net.layers(poolLyr1).inputs{1} = 'DynImg' ;
%     net.params(end-1).value = 0.01 * randn(1,1,10,1,'single');
    net.params(end-1).value = 0.1 * ones(1,1,10,1,'single');
    net.params(end).value = zeros(1,1,'single');    
    
    net.params(end-1).learningRate = 0.1 ;
    net.params(end).learningRate = 0.2 ;
  case 'none'
    
  otherwise
    error('Unknown pool type %s', opts.pool1Type) ;
end



% second pool layer (max pooling)
poolLyr2 = find(arrayfun(@(a) strcmp(a.name, opts.pool2Layer), net.layers)==1);
net.addLayer('tempPoolMax',TemporalPooling('method','max'),...
  {net.layers(poolLyr2(1)).inputs{1},'VideoId2'},'tempPoolMax');

net.layers(poolLyr2).inputs{1} = 'tempPoolMax';

% add multi-class error
net.addLayer('errMC',ErrorMultiClass(),{'prediction','label'},'mcerr');

net_ = net.saveobj ;
net = dagnn.DagNN.loadobj(net_) ;

net.removeLayer('loss') ;
net.addLayer('loss', ...
             LossNormalized('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;
           
% replace standard matconvnet bnorm with my version
bns = find(arrayfun(@(a) strcmp(class(a.block), 'dagnn.BatchNorm'), net.layers)==1);
for i=1:numel(bns)
  bb = net.layers(bns(i)).block ;
  net.layers(bns(i)).block = BatchNormN('numChannels',bb.numChannels,...
  'epsilon',bb.epsilon,...
  'opts',bb.opts) ;
end
