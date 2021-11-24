function [nRowsToSubplot,nColsToSubplot] = goodSubPlotRowCols(nSubplot)
switch nSubplot
    case 0
        nRows_nCols = [1 1];
    case 1
        nRows_nCols = [1 1];
    case 2
        nRows_nCols = [1 2];
    case 3
        nRows_nCols = [1 3];
    case 4
        nRows_nCols = [2 2];
    case 5
        nRows_nCols = [2 3];
    case 6
        nRows_nCols = [2 3];
    case 7
        nRows_nCols = [2 4];
    case 8
        nRows_nCols = [2 4];
    case 9
        nRows_nCols = [3 3];
    case 10
        nRows_nCols = [3 4];
    case 11
        nRows_nCols = [3 4];
    case 12
        nRows_nCols = [3 4];        
    case 13
        nRows_nCols = [3 5];        
    case 14
        nRows_nCols = [3 5];                
    case 15
        nRows_nCols = [3 5];
    otherwise
        nRows_nCols = [ceil(sqrt(nSubplot)) round(sqrt(nSubplot))];%%% will always work see bottom %% nRows_nCols = [ceil(sqrt(nSubplot)) ceil(sqrt(nSubplot))];
end
nRowsToSubplot = nRows_nCols(1);
nColsToSubplot = nRows_nCols(2);


% for n = 1:10000
%     a(n) = ceil(sqrt(n))* round(sqrt(n))-n;
% end
% 
% figure(423342)
% stairs(a)