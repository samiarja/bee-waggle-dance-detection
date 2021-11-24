function [ output_args ] = setFont( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
try
set(gca,'fontsize',input_args)
catch
    oops = 1
end
end

