function y = vl_nnl2norm(x,param,dzdy)
% author: Hakan Bilen
% l2 normalize whole feature map

sc = param(1);
clip = param(2:3);
offset = param(4);

if nargin == 3
  assert(all(size(x) == size(dzdy)));
else
  dzdy = [];
end

x_sz = size(x);
if ~all(x_sz([1 2]) == 1)
  % Create an array of size #channels x #samples
  x = reshape(x, prod(x_sz(1:3)), []);
end


x = x + offset;

if isempty(dzdy)
 
  y = (bsxfun(@times, x, sc./(sqrt(sum(x .* x)) + single(1e-12))));
  % clip max values
  if all(y(:)<clip(1) | y(:)>clip(2))
    warning('Too small clipping interval');
    fprintf('min %f max %f\n',min(y(:)),max(y(:)));
  end
  
  y(y(:)<clip(1)) = clip(1);
  y(y(:)>clip(2)) = clip(2);
  
  
else
  if ~all(x_sz([1 2]) == 1)
    dzdy = reshape(dzdy, prod(x_sz(1:3)), []);
  end
  
  len_ = 1./sqrt(sum(x.*x)+single(1e-12));
  dzdy_ = bsxfun(@times,dzdy,len_.^3);
  y = sc * (bsxfun(@times,dzdy,len_)-bsxfun(@times,x,sum(x.*dzdy_)));
end

if ~all(x_sz([1 2]) == 1)
  y = reshape(y, x_sz);
end
% 
% if isempty(dzdy)
%   fprintf(' fwd-l2 %.2f ',sqrt(sum(y(:).^2)));
% else
%   fprintf(' back-l2 %f dzdy %f ',sqrt(sum(y(:).^2)),sqrt(sum(dzdy(:).^2)));
% end
