function itr_matrix = itr(acc_matrix,M,t)
% Information Transfer Rate (ITR) [1] calculation.

% Input: -acc_matrix: Accuracy matrix
%        -M: # of characters
%        -t: Total signal length (visual cue + desired signal length)

% Output: -ITR = (log2(M) + p*log2(p) + (1-p)*log2((1-p)/(M-1)))*(60/t)
%              = (log2(M)*(60/t) when p=1
%              = 0 when p<1/M

% [1] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and
% T. M. Vaughan, “Brain–computer interfaces for communication
% and control,” Clinical Neurophysiology, vol. 113, no. 6, pp. 767–791, 2002.
size_mat = size(acc_matrix);
itr_matrix = zeros(size_mat);
for i=1:size_mat(1)
    for j=1:size_mat(2)
        p= acc_matrix(i,j);
        if  p < 1/M
            itr_matrix(i,j)=0;
        elseif p == 1
            itr_matrix(i,j)= log2(M)*(60/t);
        else
            itr_matrix(i,j)= (log2(M) + p*log2(p) + (1-p)*log2((1-p)/(M-1)))*(60/t);
        end
        
    end
end
end
