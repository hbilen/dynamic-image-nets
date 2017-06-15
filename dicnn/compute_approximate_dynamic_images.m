function di = compute_approximate_dynamic_images(images)
% Computes approximate dynamic images for a given array of images
% IMAGES must be a tensor of H x W x D x N dimensionality or
% cell of image names

% For the exact dynamic images, use the code
% http://users.cecs.anu.edu.au/~basura/dynamic_images/code.zip
% Explained here http://arxiv.org/abs/1512.01848

if isempty(images)
  di = [] ;
  return ;
end


if iscell(images)
  imagesA = cell(1,numel(images)) ; 
  for i=1:numel(images)
    if ~ischar(images{i})
      error('images must be an array of images or cell of image names') ;
    end
    imagesA{i} = imread(images{i}) ;
  end
  images = cat(4,imagesA{:}) ;
end

N = size(images,4) ;
di = vl_nnarpooltemporal(single(images),ones(1,N)) ;


