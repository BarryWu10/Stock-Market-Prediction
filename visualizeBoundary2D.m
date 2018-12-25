function visualizeBoundary2D(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it
%  Modified version of AndrewNg visualizeboundary to work with libsvm

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   y = ones(length(this_X),1);
   vals(:, i) = svmpredict(y, this_X, model,'-q');
end

% Plot the SVM boundary
hold on
%contour(X1, X2, vals, [0 0], 'Color', 'b','linewidth',2);
contour(X1, X2, vals, [0.5 0.5], 'Color', 'b','linewidth',2);
hold off;

end
