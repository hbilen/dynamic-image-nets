classdef AppRankPooling < dagnn.ElementWise
  % author: Hakan Bilen
  % dagnn wrapper for approximate rank pooling
  
 
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnarpooltemporal(inputs{1},inputs{2});
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2);
      derInputs{1} = vl_nnarpooltemporal(inputs{1},inputs{2},derOutputs{1});
      derParams = {} ;
    end
    
    function obj = AppRankPooling(varargin)
      obj.load(varargin) ;  
    end  
    
  end
end

