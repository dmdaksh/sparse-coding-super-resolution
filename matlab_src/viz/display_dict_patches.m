
% display dictionary patches as separate squares in a big matrix

addpath('utils'); %/matlab_src/utils, contains py2mat.m
pickle = py.importlib.import_module('pickle');

% %[ dict params
% dict_sizes = [512 1024 2048];
% dict_patchsizes = [3 5];
% dict_lambda = 0.1;  dict_lambda_str = num2str(dict_lambda);
% dict_zoom = 3;      dict_zoom_str = num2str(dict_zoom);


%[ dict files
dh_files = {
    'Dh_1024_US3_L0.1_PS5.pkl',...	
    'Dh_2048_US3_L0.1_PS3.pkl'	,...
    'Dh_2048_US3_L0.1_PS5.pkl',...
    'Dh_512_US3_L0.1_PS5.pkl'
    };
dl_files = strrep(dh_files,'Dh','Dl');

%[ dict dir
dict_dir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/dicts_bkp';

%% plot by manually-defined file names
%[ load dictionaries (after training..)
for dictind=3 %4 %1:length(dh_files)
    dh_path = fullfile(dict_dir,dh_files{dictind});
    dl_path = fullfile(dict_dir,dl_files{dictind});
        
    [~,~,fext] = fileparts(dl_path);
    switch fext
        case '.pkl'
            D = loadDictionaryPkl(pkl_file);
            D = loadDictionaryPkl(pkl_file);

%             %[load Dl pkl, convert to mat
%             fh = py.open(dl_path, 'rb');
%             P = pickle.load(fh);    % pickle file loaded to Python variable
%             fh.close();
%             Dl = py2mat(P);         % pickle data converted to MATLAB native variable
%             %[load Dl pkl, convert to mat
%             fh = py.open(dh_path, 'rb');
%             P = pickle.load(fh);    % pickle file loaded to Python variable
%             fh.close();
%             Dh = py2mat(P);         % pickle data converted to MATLAB native variable
        case '.mat'
            load(dict_path,'Dh','Dl');
    end
    
    fprintf(' -making plots..\n');
    %[ get patch size
    patch_size = str2double(extractBefore(extractAfter(dh_path,'_PS'),'.pkl'));
    %[ split Dl into 4 features
    [mh,nh] = size(Dh);
    Dl_feat = permute(reshape(Dl,mh,[],nh),[1 3 2]);
    
    
    %: display dictionary atoms

    %% 1x5 subplots
    %[ create figure+subplots
    figure('position',[248 356 1222 510],'color','w');
    tiledlayout(1,5);

    %[ plot Dh
    display_network_nonsquare_subplots(Dh,patch_size);
    title('$D_h$ patches','FontSize',25,'Interpreter','latex');

    %[ plot Dl
    figure('position',[462 176 616 546],'color','w');
    tiledlayout(1,1);
    display_network_nonsquare_subplots(Dl,patch_size*2);
    title('$D_{l}$ patches','FontSize',25,'Interpreter','latex');

    %[ plot the 4 Dl features separately
    display_network_nonsquare_subplots(Dl_feat(:,:,1),patch_size);
    title('$D_{l,1}$ patches','FontSize',25,'Interpreter','latex');
    display_network_nonsquare_subplots(Dl_feat(:,:,2),patch_size);
    title('$D_{l,2}$ patches','FontSize',25,'Interpreter','latex');
    display_network_nonsquare_subplots(Dl_feat(:,:,3),patch_size);
    title('$D_{l,3}$ patches','FontSize',25,'Interpreter','latex');
    display_network_nonsquare_subplots(Dl_feat(:,:,4),patch_size);
    title('$D_{l,4}$ patches','FontSize',25,'Interpreter','latex');

    drawnow;
    fig_name = ['show-atoms_Dh,Dl_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
        '_zoom-' num2str(upscaleFactor) '.png'];
    fig_path = fullfile(outdir,fig_name);
    print(gcf,fig_path,'-dpng');

    %% indv figs
    %[ create figure+subplots
    figure('position',[462 176 616 546],'color','w');
    tiledlayout(1,1);
    %[ plot Dh
    display_network_nonsquare_subplots(Dh,patch_size);
    title('$D_h$ patches','FontSize',25,'Interpreter','latex');

    figure('position',[462 176 616 546],'color','w');
    tiledlayout(1,1);
    %[ plot Dh
    display_network_nonsquare_subplots(Dl_feat(:,:,1),patch_size);
    title('$D_{l,1}$ patches','FontSize',25,'Interpreter','latex');

    figure('position',[462 176 616 546],'color','w');
    tiledlayout(1,1);
    %[ plot Dh
    display_network_nonsquare_subplots(Dl_feat(:,:,2),patch_size);
    title('$D_{l,2}$ patches','FontSize',25,'Interpreter','latex');

    figure('position',[462 176 616 546],'color','w');
    tiledlayout(1,1);
    %[ plot Dh
    display_network_nonsquare_subplots(Dl_feat(:,:,3),patch_size);
    title('$D_{l,3}$ patches','FontSize',25,'Interpreter','latex');

    figure('position',[462 176 616 546],'color','w');
    tiledlayout(1,1);
    %[ plot Dh
    display_network_nonsquare_subplots(Dl_feat(:,:,4),patch_size);
    title('$D_{l,4}$ patches','FontSize',25,'Interpreter','latex');

%     drawnow;
%     fig_name = ['show-atoms_Dh,Dl_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
%         '_zoom-' num2str(upscaleFactor) '.png'];
%     fig_path = fullfile(outdir,fig_name);
%     print(gcf,fig_path,'-dpng');
    

end


% %[ separate 4 features from each patch of low-res dict
% Dl_sep = reshape(Dl,size(Dh,1),[]);
% %[ display dictionary atoms
% figure('position',[248 356 1222 510],'color','w');
% tiledlayout(1,2);
% display_network_nonsquare_subplots(Dl_sep,patch_size);
% title('$D_l$ patches','FontSize',25,'Interpreter','latex');
% display_network_nonsquare_subplots(Dh,patch_size);
% title('$D_h$ patches','FontSize',25,'Interpreter','latex');
% drawnow;
% fig_name = ['show-atoms_Dh,Dl-sep_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
%     '_zoom-' num2str(upscaleFactor) '.png'];
% fig_path = fullfile(outdir,fig_name);
% print(gcf,fig_path,'-dpng');

%% plot by looping params

% %[ load dictionaries (after training..)
% for sizeind=1:length(dict_sizes)
%     dict_size = dict_sizes(sizeind);
%     for patchind=1:length(dict_patchsizes)
%         dict_patchsize = dict_patchsizes(patchind);
%         
%         dl_name = strcat('Dl_',num2str(dict_size),'_US',dict_zoom_str, ...
%             '_L',dict_lambda_str,'_PS',num2str(dict_patchsize),'.pkl');
%         dh_name = strcat('Dh_',num2str(dict_size),'_US',dict_zoom_str, ...
%             '_L',dict_lambda_str,'_PS',num2str(dict_patchsize),'.pkl');
% 
%         dl_path = fullfile(dict_dir,dl_name);
%         dh_path = fullfile(dict_dir,dh_name);
%         
%         [~,~,fext] = fileparts(dl_path);
%         switch fext
%             case '.pkl'
%                 
%                 %[load Dl pkl, convert to mat
%                 fh = py.open(dl_name, 'rb');
%                 P = pickle.load(fh);    % pickle file loaded to Python variable
%                 fh.close();
%                 Dl = py2mat(P);         % pickle data converted to MATLAB native variable
%                 %[load Dl pkl, convert to mat
%                 fh = py.open(dl_name, 'rb');
%                 P = pickle.load(fh);    % pickle file loaded to Python variable
%                 fh.close();
%                 Dl = py2mat(P);         % pickle data converted to MATLAB native variable
%             case '.mat'
%                 load(dict_path,'Dh','Dl');
%         end
%         
%         fprintf(' -making plots..\n');
%         
%         %[ display dictionary atoms
%         figure('position',[248 356 1222 510],'color','w');
%         tiledlayout(1,2);
%         display_network_nonsquare_subplots(Dl,patch_size*2);
%         title('$D_l$ patches','FontSize',25,'Interpreter','latex');
%         display_network_nonsquare_subplots(Dh,patch_size);
%         title('$D_h$ patches','FontSize',25,'Interpreter','latex');
%         drawnow;
%         fig_name = ['show-atoms_Dh,Dl_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
%             '_zoom-' num2str(upscaleFactor) '.png'];
%         fig_path = fullfile(outdir,fig_name);
%         print(gcf,fig_path,'-dpng');
%         
%         %[ separate 4 features from each patch of low-res dict
%         Dl_sep = reshape(Dl,size(Dh,1),[]);
%         %[ display dictionary atoms
%         figure('position',[248 356 1222 510],'color','w');
%         tiledlayout(1,2);
%         display_network_nonsquare_subplots(Dl_sep,patch_size);
%         title('$D_l$ patches','FontSize',25,'Interpreter','latex');
%         display_network_nonsquare_subplots(Dh,patch_size);
%         title('$D_h$ patches','FontSize',25,'Interpreter','latex');
%         drawnow;
%         fig_name = ['show-atoms_Dh,Dl-sep_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
%             '_zoom-' num2str(upscaleFactor) '.png'];
%         fig_path = fullfile(outdir,fig_name);
%         print(gcf,fig_path,'-dpng');
%     
%     end
% end

%% analysis

%     %[ Analyze dict training results
%     res = load('NewDictionary/reg_s_c_stat_512_0.15_5_s2.mat');
%     niters = length(res.regscstat.fobj_avg);
%     
%     %[ plot dict training results/stats
%     figure('color','w','position',[182 492 1282 244]);
%     tiledlayout(1,4);
%     
%     nexttile
%     p1 = plot(1:niters, res.regscstat.fobj_avg,'LineWidth',1.5,'Marker','^');
%     title('DL Objective Function');
%     xlabel('Iteration');
%     ylabel('Objective value');
%     set(gca,'FontSize',12);
%     
%     nexttile
%     p2 = plot(1:niters, 100*res.regscstat.sparsity,'LineWidth',1.5,'Marker','^');
%     title('Codebook coefficient sparsity');
%     xlabel('Iteration');
%     ylabel('Sparsity level (% nonzero)');
%     set(gca,'FontSize',12);
%     
%     nexttile
%     hold on;
%     p3a = plot(1:niters, res.regscstat.stime,'LineWidth',1.5,'Marker','^');
%     p3b = plot(1:niters, res.regscstat.btime,'LineWidth',1.5,'Marker','v');
%     title('Computation time');
%     xlabel('Iteration');
%     ylabel('Elapsed time (s)');
%     legend('Sparse code update','Dictionary update', ...
%         'location','best','fontsize',12);
%     set(gca,'FontSize',12);
%     
%     nexttile
%     p4 = plot(1:niters, cumsum(res.regscstat.elapsed_time),'LineWidth',1.5,'Marker','v');
%     title({'Cumulative elapsed time',sprintf('Total time = %.1f min',sum(res.regscstat.elapsed_time)/60)});
%     xlabel('Iteration');
%     ylabel('Elapsed time (s)');
%     set(gca,'FontSize',12);
%     
%     print(gcf,'NewDictionary/DictTrain-Testing-Results_plots.png','-dpng');


