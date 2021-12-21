function pvalues = CE_method(nbits, npar, popsize, ff, nelite, ...
    max_iter, plot_iter, print_final, print_final_seq, verbose)
% Cross entropy method
% INPUTS: 
%     nbits - length of each code sequence
%     npar - number sequences in code family
%     popsize - number of points sampled from pmf at each iteration
%     ff - cost function to use for evaluting points
%     num_elite - number of top performing points from pop to update pmf
%     max_iter - maximum number of iterations
%     plot_iter - true/false whether to plot itermediate steps
%     print_final - true/false whether to print final results
%     print_final_seq - true/false whether to print final sequence/corrs
%     verbose - true/false whether to print progress during optimization
% OUTPUTS: 
%     pvalues - final values for bernoulli random process

% global pop
global iter_plotstep
global xlabel_str
global ylabel_str
global tag_on_str
global title_str

totlength = nbits*npar;
pvalues = 0.5*ones(1, totlength);


% Get plot handles
if plot_iter
    plot_handle = figure();   
    auto_plot_handle = figure();     cross_plot_handle = figure();
end
    
disp(['nbits = ', num2str(nbits), ', npar = ', num2str(npar), ...
    ', popsize = ', num2str(popsize), ', nelite = ', num2str(nelite)]);

% Number of iterations of CE method
% tot_iter = 50;
converged = false;
i = 0;
while i < max_iter && ~converged
    % Update population
    pop = rand_bernoulli(pvalues, popsize);
    % Initialize cost and other items to set up the main loop
    [cost, auto_comp, cross_comp] = ff(pop);

    % Get elite population
    [~, sorted_i] = sort(cost);
    elite_indiv = pop(sorted_i(1:nelite),:);
    
    % Plot
    if plot_iter && mod(i,iter_plotstep)==0
        figure(plot_handle);
        hold on;
        grid on;
        plot(i*ones(length(cost)), cost, '.k');
        [min_cost, min_cost_i] = min(cost);
        plot(i, min_cost, '*g');
        xlabel(xlabel_str);
        ylabel(ylabel_str);
        title(title_str, 'Interpreter', 'none');
        grid on;
        pause(0.01);

        % Evaluate auto- and cross-correlation objectives separately
        [min_auto, min_auto_i] = min(auto_comp);
        [min_cross, min_cross_i] = min(cross_comp);

        figure(auto_plot_handle);
        plot(i*ones(size(auto_comp)), auto_comp, '.b'); hold on;
        plot(i, min_auto, '*m');
        plot(i, auto_comp(min_cost_i), '*g');
        plot(i, auto_comp(min_cross_i), '*r');
        xlabel(xlabel_str);
        ylabel('Mean Squared Auto-Correlation');
        title(['Evolution of Auto-Corr. ', tag_on_str], 'Interpreter', 'none');
        grid on;
        pause(0.01);

        figure(cross_plot_handle);
        plot(i*ones(size(cross_comp)), cross_comp, '.b'); hold on;
        plot(i, min_cross, '*m');
        plot(i, cross_comp(min_cost_i), '*g');
        plot(i, cross_comp(min_auto_i), '*r');
        xlabel(xlabel_str);
        ylabel('Mean Squared Cross-Correlation');
        title(['Evolution of Cross-Corr. ', tag_on_str], 'Interpreter', 'none');
        grid on;
        pause(0.01);
    end
    
    % Update distribution
    pvalues = mean(elite_indiv, 1);
    
    % Print updated version
    if verbose
        [cost, ~, ~] = ff(round(pvalues));
        disp(['Iteration ', num2str(i), ': ', num2str(cost)]);
        disp(' ');
    end
    
    % Re-check convergence
    converged = sum(pvalues == round(pvalues)) == totlength;
    i = i + 1;   % update i
    
end

%[auto_corr, cross_corr] = get_auto_and_cross_corr( round(pvalues), false );
[final_cost, final_auto, final_cross] = ff(round(pvalues));        

if print_final
    if converged
        disp(' ');
        disp(['Converged! Completed ', num2str(i-1), ' iterations.']) ;
        disp(['Final cost: ', num2str(final_cost)]);
        disp(['    mean sqr auto: ', num2str(final_auto)]);
        disp(['   mean sqr cross: ', num2str(final_cross)]);
        disp(' ');
        if print_final_seq
            disp('Final sequence: ');
            disp(pvalues);
            disp(' ');
        end
    else
        disp(' ');
        disp(['Not converged after ', num2str(max_iter), ' iterations']);
        disp(['Final cost: ', num2str(final_cost)]);
        disp(['    mean sqr auto: ', num2str(final_auto)]);
        disp(['   mean sqr cross: ', num2str(final_cross)]);
        disp(' ');
        if print_final_seq
            disp('Final p-values: ');
            disp(pvalues);
            disp(' ');
        end
    end
end