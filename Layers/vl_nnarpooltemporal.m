function Y = vl_nnarpooltemporal(X,ids,dzdy)
% author: Hakan Bilen
% approximate rank pooling
% ids indicates frame-video association (must be in range [1-N])

sz = size(X);
forward = logical(nargin<3);

if numel(ids)~=size(X,4)
  error('Error: ids dimension does not match with X!');
end

nVideos = max(ids);

if forward
  Y = zeros([sz(1:3),nVideos],'like',X);
else
  Y = zeros(size(X),'like',X);
end

for v=1:nVideos
  % pool among frames
  indv = find(ids==v);
  if isempty(indv)
    error('Error: No frames in video %d',v);
  end
  N = numel(indv);
  % magic numbers
  fw = zeros(1,N);
  if N==1
    fw = 1;
  else
    for i=1:N
      fw(i) = sum((2*(i:N)-N-1) ./ (i:N));
    end
  end
  
  if forward
    Y(:,:,:,v) =  sum(bsxfun(@times,X(:,:,:,indv),...
      reshape(single(fw),[1 1 1 numel(indv)])),4);    
  else
    Y(:,:,:,indv) = (bsxfun(@times,repmat(dzdy(:,:,:,v),[1,1,1,numel(indv)]),...
      reshape(fw,[1 1 1 numel(indv)]))) ;
  end
end
%
% if forward
  %   fprintf(' fwd-arpool %.2f ',sqrt(sum(Y(:).^2)));
  % else
  %   fprintf(' back-arpool %f ',sqrt(sum(Y(:).^2)));
% end

