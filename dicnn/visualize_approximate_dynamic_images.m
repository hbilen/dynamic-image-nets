function visualize_approximate_dynamic_images(images)
% VISUALIZE_DYNAMIC_IMAGES

di = compute_approximate_dynamic_images(images) ;

di = di - min(di(:)) ;
di = 255 * di ./ max(di(:)) ;
image(uint8(di)) ;
