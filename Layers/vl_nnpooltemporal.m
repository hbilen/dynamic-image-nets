function Y = vl_nnpooltemporal(X,ids,method,dzdy)
% author: Hakan Bilen
% temporal pooling along frames
% ids indicates frame-video association
% method 'max' or 'avg'

sz = size(X);
forward = logical(nargin<4);
Xp = permute(X,[4,1,2,3]);

if numel(ids)~=size(X,4)
  error('Error: ids dimension does not match with X!');
end

nVideos = max(ids);

if forward
  Yp = zeros([nVideos,sz(1:3)],'like',X);
  for v=1:nVideos
    % pool among frames
    indv = find(ids==v);
    Yp(v,:,:,:) = vl_nnpool(Xp(indv,:,:,:), [numel(indv),1], ...
      'pad', 0, 'stride', [numel(indv),1], 'method', method) ;
  end
else
  dzdyp = permute(dzdy,[4,1,2,3]);
  Yp = zeros(size(Xp),'like',Xp);
  for v=1:nVideos
    % pool among frames
    indv = find(ids==v);
    Yp(indv,:,:,:) = vl_nnpool(Xp(indv,:,:,:), [numel(indv),1], dzdyp(v,:,:,:), ...
      'pad', 0, 'stride', [numel(indv),1], 'method', method) ;
  end
  
end
% permute back
Y = permute(Yp,[2,3,4,1]);

% if forward
%   fprintf(' fwd-ptemp %.2f ',sqrt(sum(Y(:).^2)));
% else
%   fprintf(' back-ptemp %.2f ',sqrt(sum(Y(:).^2)));
% end
