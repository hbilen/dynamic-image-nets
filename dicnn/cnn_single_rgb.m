  function [net, info] = cnn_single_rgb(varargin)
%CNN_SINGLE_RGB Demonstrates fine-tuning a pre-trained CNN with static 
% RGB frames (SI in pami journal) on UCF101 dataset


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;

addpath Layers Datasets

opts.dataDir = fullfile('data','UCF101') ;
opts.expDir  = fullfile('exp', 'UCF101') ;
opts.modelPath = fullfile('models','resnext_50_32x4d-pt-mcn.mat');
opts.datasetFn = @cnn_ucf101_setup_data ;
opts.networkFn = @cnn_init_resnext ;
opts.pool1Type = 'none' ;
opts.pool1Layer = 'conv1' ;
opts.pool2Layer = '' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 8 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb-rgb.mat');
opts.ARPoolLayer = 'conv0'; % before conv1
opts.DropOutRate = 0.5 ;
opts.epochFactor = 5 ;

opts.split = 1; % data split
opts.reverseDyn = 0; % reverse video frames e.g.[N:-1:1]
opts.train = struct() ;
opts.train.gpus = [];
opts.train.batchSize = 128 ;
opts.train.numSubBatches = 16 ;
opts.train.solver = [] ;
opts.train.prefetch = true ;
opts.train.numEpochs = 30 ;
% resnet50
opts.train.learningRate = 1e-2 * [ones(1,2), 0.1*ones(1,1), 0.01*ones(1,1)];
% caffe-ref
opts.train.learningRate = 1e-4 * [ones(1,2), 0.1*ones(1,1), 0.01*ones(1,1)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
% opts.train.numEpochs = numel(opts.train.learningRate);

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = opts.datasetFn('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% UCF101 has 3 data splits
if opts.split>3
  error('split should be <=3');
end
imdb.images.set = imdb.images.sets(opts.split,:);

% reverse frame order
if opts.reverseDyn
  for i=1:numel(imdb.images.names)
    imdb.images.names{i} = imdb.images.names{i}(end:-1:1);
  end
end

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
net = load(opts.modelPath);
if isfield(net,'net')
  net = net.net;
end
opts.nCls = max(imdb.images.label) ;
net = opts.networkFn(net,opts);

if numel(net.meta.normalization.averageImage)>3
  sz = size(net.meta.normalization.averageImage) ;
  net.meta.normalization.averageImage = ...
    mean(reshape(net.meta.normalization.averageImage,[sz(1)*sz(2) sz(3)]),1) ;
end

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
if opts.epochFactor>0
  opts.train.train = repmat(find(imdb.images.set==1),[1 opts.epochFactor]) ;
else
  opts.train.train = NaN ;
end
opts.train.val = find(imdb.images.set==3) ;

[net, info] = cnn_train_dicnn_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                          Report accuracy
% -------------------------------------------------------------------------
errlayer = net.getLayerIndex('errMC') ;

if ~isnan(errlayer)
  cats = imdb.classes.name ;
  accs = net.layers(errlayer).block.accuracy ; 
  
  if numel(cats)~=numel(accs)
    error('wrong number of classes\n') ;
  end
  
  for i=1:numel(cats)
    fprintf('%s acc %.1f\n',cats{i},100*accs(i)) ;
  end
  fprintf('Mean accuracy %.1f\n',100*mean(accs)) ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
if isfield(meta.normalization,'border')
  bopts.border = meta.normalization.border ;  
else
  bopts.border = meta.normalization.imageSize(1:2) ./ ...
    meta.normalization.cropSize - meta.normalization.imageSize(1:2);

end

% bopts.averageImage = []; 
bopts.averageImage = meta.normalization.averageImage ;
bopts.interpolation = meta.normalization.interpolation ;
bopts.keepAspect = meta.normalization.keepAspect ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;


fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;



% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------

% batch refers to videos (not for frames)
if isempty(batch)
  inputs = {'input', [], 'label', [], 'VideoId1', [], 'VideoId2', []};
  return;
end

isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

% if ~isVal, transformation='stretch'; else transformation='none';end
if ~isVal, transformation='multiScaleRegular'; else transformation='none';end

names = imdb.images.names(batch);


% images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;

namesM = {};
nVids = numel(batch);

VideoId1 = [];
VideoId2 = [];

% step-size
stepSize = 6;
% pool nFrames into a dynamic image
nFrames = 1;
% number of dynamic images to be max pooled later
nDynImgs = 10;


c1 = 1;
for v=1:nVids
  
  name = names{v};
  nFrms = numel(name);

  nSample = nFrames;
  nr = numel(1:stepSize:nFrms);
  
  % jitter by removing 50 % and limit a batch to nMaxs * nSamples images
  if nr > 1 && (~isVal && nr>nDynImgs)
    rat = min(nDynImgs,ceil(0.50*nr));
    ri = randperm(nr);
    ri = ri(1:rat);
    r = zeros(1,nr);
    r(ri) = 1;
  else
    r = ones(1,nr);
  end
  
  c3 = 1;
  c2 = 0;
  
  for f=1:stepSize:nFrms
    if r(c3)
      idx = f:min(f+nSample-1,nFrms) ;
      if numel(idx)<nFrames
        idx = [idx idx(end) * ones(1,nFrames-numel(idx))];
      end
      namesM{end+1} = name(idx);
      VideoId1 = [VideoId1 c1 * ones(1,numel(idx))];
      c1 = c1 + 1;
      c2 = c2 + 1;
    end
    c3 = c3 + 1;
  end
  VideoId2 = [VideoId2 v * ones(1,c2) ] ;
end

images = strcat([imdb.imageDir filesep], horzcat(namesM{:}) ) ;

im = cnn_video_rgb_get_batch(images, VideoId1, opts, ...
  'transformation', transformation, 'prefetch', nargout == 0, ...
  'subMean', false) ;

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', imdb.images.label(batch), ...
    'VideoId2', VideoId2};
end
