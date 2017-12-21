function [net, info] = cnn_dicnn_of(varargin)
%CNN_DICNN_OF Fine-tunes a pre-trained CNN with dynamic images on optical
% (DOF in pami journal) flow frames on UCF101 dataset


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matconvnet', 'contrib', 'mcnExtraLayers', 'setup_mcnExtraLayers.m')) ;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matconvnet', 'contrib', 'autonn', 'setup_autonn.m')) ;

addpath Layers Datasets

opts.dataDir = fullfile('data','UCF101') ;
opts.expDir  = fullfile('exp', 'UCF101') ;
opts.modelPath = fullfile('models','resnext_50_32x4d-pt-mcn.mat.mat') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 8 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb-of.mat');
opts.pool1Layer = 'conv0'; % before conv1
opts.pool1Type = 'arpool'; % before conv1
opts.pool2Layer = 'fc6'; % before conv1
opts.DropOutRate = 0.85 ;
opts.datasetFn = @cnn_ucf101_of_setup_data ;
opts.networkFn = @cnn_init_resnext ;
opts.network = [] ;

opts.split = 1; % data split
opts.reverseDyn = 0; % reverse video frames e.g.[N:-1:1]
opts.numDynImgs = 10 ;
opts.epochFactor = 5 ;

opts.train = struct() ;
opts.train.gpus = [];
opts.train.batchSize = 128 ;
opts.train.numSubBatches = 32 ;
opts.train.solver = [] ;
opts.train.prefetch = true ;
opts.train.learningRate = 1e-2 ;
opts.train.numEpochs = 30 ;
% opts.train.savePreds = true ;
opts.train.randomSeed = 0 ;

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


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
if isempty(opts.network)
  net = load(opts.modelPath);
  if isfield(net,'net')
    net = net.net;
  end
  opts.nCls = max(imdb.images.label) ;
  % net = dagnn.DagNN.loadobj(net) ;
  net = opts.networkFn(net,opts) ;
  
  % two channels instead of 3 RGB
  net.params(1).value = net.params(1).value(:,:,1:2,:) ;
  
  % Set the class names in the network
  net.meta.classes.name = imdb.classes.name ;
  net.meta.classes.description = imdb.classes.name ;
else
  assert(isa(opts.network,'dagnn.DagNN')) ;
  net = opts.network ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
if opts.epochFactor>0
  opts.train.train = repmat(find(imdb.images.set==1),[1 opts.epochFactor]) ;
else
  opts.train.train = NaN ;
  opts.train.numEpochs = 1 ;
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

bopts.averageImage = 128 * ones([1 1 2],'single') ;
bopts.numDynImgs = opts.numDynImgs ;

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
nFrames = 10;
% number of dynamic images to be max pooled later
nDynImgs = opts.numDynImgs ;
opts = rmfield(opts,'numDynImgs') ;


c1 = 1;
for v=1:nVids
  
  name = names{v};
  nFrms = numel(name)/2;

  nSample = nFrames;
  
  if isVal
    startF = 1 ;
  else
    startF = ceil(stepSize/2) ;
  end
  nr = numel(startF:stepSize:nFrms);
  
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
  
  for f=startF:stepSize:nFrms
    if r(c3)
      idx = f:min(f+nSample-1,nFrms) ;
      if numel(idx)<nFrames
        idx = [idx idx(end) * ones(1,nFrames-numel(idx))];
      end
      idxu = 2*idx - 1;
      idxv = 2*idx;
      idxuv = zeros(1,2 * numel(idxu)) ;
      idxuv(1:2:end) = idxu ;
      idxuv(2:2:end) = idxv ;
            
      namesM{end+1} = name(idxuv);
      VideoId1 = [VideoId1 c1 * ones(1,numel(idx))];
      c1 = c1 + 1;
      c2 = c2 + 1;
    end
    c3 = c3 + 1;
  end
  VideoId2 = [VideoId2 v * ones(1,c2) ] ;
end

images = strcat([imdb.imageDir filesep], horzcat(namesM{:}) ) ;

im = cnn_video_of_get_batch(images, VideoId1, opts, ...
  'transformation', transformation, 'prefetch', nargout == 0) ;

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', imdb.images.label(batch), ...
    'VideoId1', VideoId1, 'VideoId2', VideoId2};

end
