function tderica_main(dimEmbed,numComp,subsample)

% set meta parameters
dirRaw = 'C:\Users\toyo\Desktop\tcrs4d_200218\20221118freezed\';
strHeaderRemove = {'180710h','180712e','190313g','190314e'};
frameRemove     = [     3000,     2450,     4600,     3900];
idxNormalize = 2;
numSamplesSelect = 10;


% set default parameters if not input
if ~exist('dimEmbed','var')||isempty(dimEmbed)
    dimEmbed = 300;
end
if ~exist('numComp','var')||isempty(numComp)
    numComp = 14;
end
if ~exist('subsample','var')||isempty(subsample)
    subsample = 1; % sub-sampling of time course
end


isSameOrders = true; % To keep the same order as the figures in the paper
if isSameOrders
    dOrders = load('tderica_orders.mat','orderMotifs','orderCells','orderCellsSelected');
    orderMotifs = dOrders.orderMotifs;
    orderCells = dOrders.orderCells;
    orderCellsSelected = dOrders.orderCellsSelected;
else
    orderMotifs = [];
    orderCells = [];
    orderCellsSelected = [];
end


%%% manage cache folder
dirCache = fullfile(fileparts(mfilename('fullpath')),'cache');
if ~exist(dirCache,'dir')
    mkdir(dirCache);
else
    % delete(fullfile(dirCache,'*')); % for clearing cache
end


%%% load and arrange the whole-brain activity of multiple samples
[tcrsNArranged,strNames] = loadData(dirRaw,strHeaderRemove,frameRemove,idxNormalize,dirCache);

%%% select several samples to remove missing values
[idxSamplesSelected,idxCellsSelected] ...
    = selectDataWithoutMissingValue(tcrsNArranged,numSamplesSelect);

%%% do tde-rica
tcrsSelected = tcrsNArranged(:,idxCellsSelected,idxSamplesSelected);
[occurrencesBase,motifsBase,tcrsRecBase] = doTdeRica(tcrsSelected,dimEmbed,numComp,subsample,dirCache);

%%% visualize results
strNamesSelected = strNames(idxCellsSelected);
[occurrencesBaseSorted,motifsBaseSorted,tcrsSelectedSorted,tcrsRecBaseSorted,...
    strNamesBaseSorted,orderCellsSelected,orderMotifs] ...
    = sortByClustering(occurrencesBase,motifsBase,tcrsSelected,tcrsRecBase,...
    strNamesSelected,orderCellsSelected,orderMotifs); %#ok<ASGLU>
fileBase = 'tderica_basic_';
makeFigures(occurrencesBaseSorted,motifsBaseSorted,tcrsSelectedSorted,tcrsRecBaseSorted,...
    strNamesBaseSorted,idxSamplesSelected,subsample,dirCache,fileBase)


%%% do matrix factorization
[occurrencesAll,motifsAll,tcrsRecAll] ...
    = doTdeRicaMatrixFactorization(tcrsNArranged,idxSamplesSelected,...
    idxCellsSelected,occurrencesBase,motifsBase,subsample,dirCache);

%%% visualize results
[occurrencesAllSorted,motifAllSorted,tcrsAllSorted,tcrsRecAllSorted,...
    strNamesAllSorted,orderCells,orderMotifs] ...
    = sortByClustering(occurrencesAll,motifsAll,tcrsNArranged,tcrsRecAll,...
    strNames,orderCells,orderMotifs);
idxSamplesAll = 1:size(tcrsNArranged,3);
fileBaseAll = 'tderica_matrixFactorization_';
makeFigures(occurrencesAllSorted,motifAllSorted,tcrsAllSorted,tcrsRecAllSorted,...
    strNamesAllSorted,idxSamplesAll,subsample,dirCache,fileBaseAll)


end


function [occurrencesAll,motifsAll,tcrsRecAll] ...
    = doTdeRicaMatrixFactorization(tcrsNArranged,idxSamplesSelected,...
    idxCellsSelected,occurrencesBase,motifsBase,subsample,dirCache)
%%% Do Matrix factorizaion based on the result of TDE-RICA to inpute the
%%% missing values
%%%
%%%
%%% Matrix factorization:
%%%
%%% X: time-deley embedded data, [numTEmbed*numSamples,numCells*dimEmbed]
%%% W: temporal acitivaion of motif, [numTEmbed*numSamples,numComp]
%%% M: W'*X, independent component or motif, [numComp,numCells*dimEmbed]
%%% X is decomposed by rica as minimize the 2-norm of (W*W'*X-X) or
%%% (W * M - X) with some regularization (independendency of M = W'X).
%%% (For algorithm details, please see
%%% https://jp.mathworks.com/help/stats/feature-extraction.html#bvmxyf6-1)
%%% Please note that usual ICA for blind source decomposition assumes that
%%% temporal patterns are independent, but our setup assumes the weights of
%%% the temporal patterns are independent, and the data matrix is
%%% transposed.
%%%
%%% Here we  expand the data matrix as [X,A; B,C] and the size of the data
%%% matrix is [numTEmbed*numSamples,numCells*dimEmbed].
%%% Please note that X does not include NaN but the others include NaN.
%%% W and M also expanded to [WBase;WAdd] and [MBase,MAdd], respectively.
%%% The decomposition will be achieved by minimizing the 2-norm of
%%% [WBase;WAdd] * [MBase,MAdd] - [X,A;B,C] with fixed WBase (=W) and Mbase
%%% (=M). Here we minimize the sum of 2-norm of WBase*MBase-X,
%%% WBase*MAdd-A, WAdd*MBase-B, and WAdd*MBase-C, but the 1st term is fixed
%%% and the other 3 terms are the targets of the optimization.
%%%

h = tic();
fprintf('Do TDE-RICA with Matrix Factorization ... ');

[numComp,dimEmbed,~] = size(motifsBase);
tcrsSubsampled = tcrsNArranged(1:subsample:end,:,:);
[maxT,numCells,numSamples] = size(tcrsSubsampled);
isSamplesSelected = false(numSamples,1);
isSamplesSelected(idxSamplesSelected) = true;
isCellsSelected = false(numCells,1);
isCellsSelected(idxCellsSelected) = true;
numTEmbed = maxT-dimEmbed+1;

pathSaveTdeRicaMatrixFactorziation = fullfile(dirCache,...
    sprintf('tderica_matrixFactorization_dimEmbed=%d_numComp=%d_subSample=%d.mat',...
    dimEmbed,numComp,subsample));

if ~exist(pathSaveTdeRicaMatrixFactorziation,'file')
    
    %%% TDE
    fprintf('calculating ... ');    

    tcrsEmbed = nan(numTEmbed,dimEmbed,numCells,numSamples);
    for p=1:numSamples
        tcrsEmbed(:,:,:,p) = delayembed(tcrsNArranged(1:subsample:end,:,p),dimEmbed);
    end
    isTValidTmp = ~any(all(isnan(tcrsEmbed),3),2);
    isTValidBase = reshape(isTValidTmp(:,:,:, isSamplesSelected),[],1);
    isTValidAdd  = reshape(isTValidTmp(:,:,:,~isSamplesSelected),[],1);
    
    %%% matrix factorization
    
    WBase = reshape(occurrencesBase,[numTEmbed*sum(isSamplesSelected),numComp]);
    WBase = WBase(~any(isnan(WBase),2),:);
    MBase = reshape(motifsBase,[numComp,dimEmbed*sum(isCellsSelected)]);
    
    A = reshape(permute(tcrsEmbed(:,:,~isCellsSelected, isSamplesSelected),[1,4,2,3]),...
        [numTEmbed*sum( isSamplesSelected),dimEmbed*sum(~isCellsSelected)]);
    A = A(isTValidBase,:);
    
    B = reshape(permute(tcrsEmbed(:,:, isCellsSelected,~isSamplesSelected),[1,4,2,3]),...
        [numTEmbed*sum(~isSamplesSelected),dimEmbed*sum( isCellsSelected)]);
    B = B(isTValidAdd,:);
    
    C = reshape(permute(tcrsEmbed(:,:,~isCellsSelected,~isSamplesSelected),[1,4,2,3]),...
        [numTEmbed*sum(~isSamplesSelected),dimEmbed*sum(~isCellsSelected)]);
    C = C(isTValidAdd,:);
    
    clear tcrsEmbed;
    
    numRowsAdd = size(B,1);
    numColsAdd = size(A,2);
    param0 = rand((numRowsAdd+numColsAdd)*numComp,1);
    
    solver  = classreg.learning.fsutils.Solver(1);
    solver.SolverName = 'lbfgs';
    solver.HaveGradient            = true;
    solver.GradientTolerance       = 1e-6;
    solver.StepTolerance           = 1e-6;
    solver.IterationLimit          = 1000;    
    solver.MaxLineSearchIterations = 50;
    solver.Verbose                 = 1;
    
    results = solver.doMinimization(@(param)objfun(param,WBase,MBase,A,B,C),param0,1);
    
    %%% parse results
    
    param = results.xHat;
    WAdd = reshape(param(1:numRowsAdd*numComp),numRowsAdd,numComp);
    MAdd = reshape(param(numRowsAdd*numComp+1:end),numComp,numColsAdd);
    
    save(pathSaveTdeRicaMatrixFactorziation,'WAdd','MAdd','isTValidAdd');
    
else
    fprintf('load from cache ...');
    load(pathSaveTdeRicaMatrixFactorziation,'WAdd','MAdd','isTValidAdd');
    
end

occurrencesAdd = nan(numel(isTValidAdd),numComp);
occurrencesAdd(isTValidAdd,:) = WAdd;
occurrencesAdd = reshape(occurrencesAdd,numTEmbed,sum(~isSamplesSelected),numComp);

occurrencesAll = nan(numTEmbed,numSamples,numComp);
occurrencesAll(:, isSamplesSelected,:) = occurrencesBase;
occurrencesAll(:,~isSamplesSelected,:) = occurrencesAdd;

motifsAdd = reshape(MAdd,numComp,dimEmbed,sum(~isCellsSelected));

motifsAll = nan(numComp,dimEmbed,numCells);
motifsAll(:,:, isCellsSelected) = motifsBase;
motifsAll(:,:,~isCellsSelected) = motifsAdd;


tcrsRecAll = nan(maxT,numCells,numSamples);
for p=1:numSamples
    tcrsRecAll(:,:,p) = delayembed_inv(reshape(permute(occurrencesAll(:,p,:),[1,3,2])...
        *reshape(motifsAll,[numComp,dimEmbed*numCells]),[numTEmbed,dimEmbed,numCells]));
end


fprintf('finished in %0.3f sec\n',toc(h));

end


function [ofv,grad] = objfun(param,WBase,MBase,A,B,C)
%%% objective function: 2-norm-of(WBase*Madd-A)
%%% gradient: r(2-norm-of(WBase*MAdd-A))/r(Madd)=2*WBase'*(WBase*Madd-A)
numComp = size(WBase,2);
numRowsAdd = size(B,1);
numColsAdd = size(A,2);
WAdd = reshape(param(1:numRowsAdd*numComp),numRowsAdd,numComp);
MAdd = reshape(param(numRowsAdd*numComp+1:end),numComp,numColsAdd);

dA = WBase * MAdd  - A;
dB = WAdd  * MBase - B;
dC = WAdd  * MAdd  - C;

dA(isnan(A)) = 0;
dB(isnan(B)) = 0;
dC(isnan(C)) = 0;

ofv = sum(dA.^2,'all') + sum(dB.^2,'all') + sum(dC.^2,'all');

if nargout>1 % grad required
    rArM = 2 * WBase' * dA;
    rBrW = 2          * dB * MBase';
    rCrM = 2 * WAdd'  * dC;
    rCrW = 2          * dC * MAdd';
    grad = cat(1,rBrW(:)+rCrW(:),rArM(:)+rCrM(:));
end

end


function [occurrencesSorted,motifsSorted,tcrsNASorted,tcrsRecSorted,...
    strNamesSorted,orderCells,orderMotifs] ...
    = sortByClustering(occurrences,motifs,tcrsNA,tcrsRec,...
    strNames,orderCells,orderMotifs)
%%% Sort cells and motifs based on clustering of motifs and occurrences.
%%% Please note that the sort order may be different between trials.

h = tic();
fprintf('Sort by clustering ... ');

[~,numSamples,numComp] = size(occurrences);

if isempty(orderCells)
    fprintf('calculating orderCells ... ');
    distance2 = squareform(pdist(reshape(permute(motifs,[3,2,1]),size(motifs,3),[])));
    orderCells = optimalleaforder(linkage(distance2),distance2);
end

if isempty(orderMotifs)
    fprintf('calculating orderMotifs ... ');
    xc = nan(numComp,numComp,numSamples);
    for p=1:numComp
        for q=1:p-1
            for r=1:10
                isNonNan = ~any(isnan(occurrences(:,r,[p,q])),[2,3]);
                xc(p,q,r) = max(abs(xcorr(occurrences(isNonNan,r,p),occurrences(isNonNan,r,q),'normalized')));
                xc(q,p,r) = xc(p,q,r);
            end
        end
    end
    distance3 = 1 - mean(xc,3);
    distance3(eye(size(distance3))==1) = 0;
    orderMotifs = optimalleaforder(linkage(distance3),distance3);
end

occurrencesSorted = occurrences(:,:,orderMotifs);
motifsSorted = motifs(orderMotifs,:,orderCells);
tcrsNASorted  = tcrsNA( :,orderCells,:);
tcrsRecSorted = tcrsRec(:,orderCells,:);
strNamesSorted = strNames(orderCells);

fprintf('finished in %0.3f sec\n',toc(h));

end


function makeFigures(occurrences,motifs,tcrs,tcrsRec,strNames,...
    idxSamples,subsample,dirCache,fileBase)

hTic = tic();
fprintf('Make figures ... ');

[~,numSamples,numComp] = size(occurrences);
[~,dimEmbed,numCells] = size(motifs);

fileParameters = sprintf('dimEmbed=%d_numComp=%d_subsample=%d',...
    dimEmbed,numComp,subsample);
fileSave = [fileBase,fileParameters];


%%% scaling by standard deviation of motifs
sdMotifs = std(motifs,[],[2,3]);
motifs = motifs./sdMotifs;
occurrences = occurrences.*permute(sdMotifs,[3,2,1]);

climMotifs = [min(motifs,[],'all'),max(motifs,[],'all')];

strNamesDisp = strNames;
strNamesDisp(2:3:end) = strcat(strNamesDisp(2:3:end),{'----------'});
strNamesDisp(3:3:end) = strcat(strNamesDisp(3:3:end),{'----------'});
strNamesDisp(3:3:end) = strcat(strNamesDisp(3:3:end),{'----------'});

%%% make figure
pathSavePs = [fileSave,'_figs.ps'];
if exist(pathSavePs,'file')
    delete(pathSavePs);
end
for p=1:numSamples
    figure('Colormap',jet(256),'PaperType','A4','PaperOrientation','landscape');
    
    ax1 = subplot(2,2,1);
    imagesc(tcrs(1:subsample:end,:,p)');
    alpha(~isnan(tcrs(1:subsample:end,:,p)')*1);
    set(gca,'YTick',1:numCells,'YTickLabel',strNamesDisp);
    
    title(sprintf('Original time course of sample #%d, dimEmbed=%d, numComp=%d, subsample=%d',...
        idxSamples(p),dimEmbed,numComp,subsample));
    cx = caxis();
    
    for r=1:numComp
        ax2(r) = subplot(2,2*numComp+2,numComp+r+1); %#ok<AGROW>
        imagesc(permute(motifs(r,:,:),[3,2,1]));
        if r==round(numComp/2)
            title('Motifs');
        end
        set(gca,'YTickLabel',[]);
        
        ax3(r) = subplot(2*numComp,2,2*numComp+2*r-1); %#ok<AGROW>
        plot(occurrences(:,p,r));
        if r==1
            title('Occurrences');
        end
        if r~=numComp
            set(gca,'XTickLabel',[]);
        end
    end
    
    ax2c = subplot(2,2*numComp+2,2*numComp+2);
    colorbar;
    set(ax2c,'visible','off');
    
    ax4 = subplot(2,2,4);  
    imagesc(tcrsRec(:,:,p)');
    alpha(~isnan(tcrsRec(:,:,p)')*1);
    set(gca,'YTick',1:numCells,'YTickLabel',strNamesDisp);
    title('Reconstructed time course by TDE-RICA');
    caxis(cx);
    colorbar;
    
    linkaxes([ax1,ax2,ax4],'y');
    linkaxes([ax1,ax3,ax4],'x');
    set([ax2,ax2c],'clim',climMotifs);
    for r=1:numComp
        ax3(r).Position([1,3]) = ax1.Position([1,3]);
    end
    
    saveas(gcf,fullfile(dirCache,[fileSave,'_sample=',num2str(idxSamples(p)),'.fig']));
    print('-dpsc',pathSavePs,'-fillpage','-append');
    
    % close;
    
end


mi = zeros(numComp,numComp,numSamples);
for p=1:numSamples
    figure('PaperType','A4','PaperOrientation','landscape','InvertHardcopy','off');
    %     set(gcf,'Renderer','painters'); % for remote environment. software openGL cannot handle alpha settings
    
    [S,AX,BigAx,H,Hax] = plotmatrix(permute(occurrences(:,p,:),[1,3,2])); %#ok<ASGLU>
    title(BigAx,sprintf(['Occurrences of sample #%d, dimEmbed=%d, numComp=%d,',...
        ' subsample=%d, red background means MI>0.9'],...
        idxSamples(p),dimEmbed,numComp,subsample));
    hold(AX,'on');
    
    vrange = zeros(2,numComp);
    edges = cell(numComp,1);
    for q=1:numComp
        vrange(:,q) = get(Hax(q),'xlim');
        edges(q) = {get(H(q),'BinEdges')};
    end
    
    hc2 = cell(numComp,numComp);
    for q=1:numComp
        for r=q+1:numComp
            hc2{q,r} = histcounts2(occurrences(:,p,r),occurrences(:,p,q),edges{r},edges{q});
            tmpPos = get(AX(q,r),'Position');
            cla(AX(q,r));
            AX(q,r) = axes('Position',tmpPos,'box','on');
            %             histogram2(AX(q,r),'XBinEdges',edges{r},'YBinEdges',edges{q},'BinCounts',hc2{q,r},...
            %                 'DisplayStyle','tile','ShowEmptyBins','on','EdgeAlpha',0);
            imagesc(AX(q,r),edges{r}([1,end]),edges{q}([1,end]),hc2{q,r}');
            hold on;
            set(AX(q,r),'xlim',vrange(:,r),'ylim',vrange(:,q),'XTick',[],'YTick',[]);
            pxy = hc2{q,r}/sum(hc2{q,r},'all');
            px = sum(pxy,2);
            py = sum(pxy,1);
            mi(q,r,p) = sum(pxy.*log2(pxy./(px*py)),'all','omitnan');
            mi(r,q,p) = mi(q,r,p);
        end
    end
    set(AX(mi(:,:,p)>0.9),'Color',[1,0.5,0.5]);
    
    saveas(gcf,fullfile(dirCache,[fileSave,'_plotmarix_sample=',num2str(idxSamples(p)),'.fig']));
    print('-dpsc',pathSavePs,'-fillpage','-append');
    
end


%%% only motifs
figure('Colormap',jet(256),'PaperType','A4','PaperOrientation','landscape');
for r=1:numComp
    ax2(r) = subplot(1,numComp+1,r);
    imagesc(permute(motifs(r,:,:),[3,2,1]));
    if r==1
        set(gca,'YTick',1:numCells,'YTickLabel',strNamesDisp);
    else
        set(gca,'YTickLabel',[]);
    end
end
ax2c = subplot(1,numComp+1,numComp+1);
colorbar;
set(ax2c,'visible','off');
set([ax2,ax2c],'clim',climMotifs);

saveas(gcf,fullfile(dirCache,[fileSave,'_motifs.fig']));
print('-dpsc',pathSavePs,'-fillpage','-append');


%%% calc cross correlation
maxlag = 500;
xcraw   = ones(maxlag*2+1,numSamples,numComp,numComp);
xcmean  = ones(maxlag*2+1,numComp,numComp);
xcstd   = ones(maxlag*2+1,numComp,numComp);
for p=1:numComp
    for q=1:numComp
        for r=1:numSamples
            isNonNan = ~any(isnan(occurrences(:,r,[p,q])),[2,3]);
            [xcraw(:,r,p,q),lags] = xcorr(occurrences(isNonNan,r,p),occurrences(isNonNan,r,q),'normalized',500);
        end
        xcmean(:,p,q) = mean(xcraw(:,:,p,q),2);
        xcstd(:,p,q)  = std(xcraw(:,:,p,q),[],2);
    end
end


%%% mean of cross correlation
figure('PaperType','A3','PaperOrientation','landscape','InvertHardcopy','off');
set(gcf,'Renderer','painters'); % for remote. software openGL cannot handle alpha settings
tmpt = -maxlag:maxlag;
for p=1:numComp
    for q=1:numComp
        subplot(numComp,numComp,(p-1)*numComp+q);
        plot(tmpt,xcmean(:,p,q),'r-',...
            tmpt,xcmean(:,p,q)+xcstd(:,p,q),'b-',...
            tmpt,xcmean(:,p,q)-xcstd(:,p,q),'b-');
        hold on;
        plot(tmpt([1,end]),[0,0],'k:');
        plot([0,0],ylim,'k:');
        %         plot([0,0],[-1,1],'k:');
        title(sprintf('%d-%d',p,q));
        if any((xcmean(:,p,q)+xcstd(:,p,q))<0) || any((xcmean(:,p,q)-xcstd(:,p,q))>0)
            set(gca,'Color',[0.8,1,0.8]);
        end
        if (p~=numComp)
            set(gca,'XTick',[])
        end
    end
end

saveas(gcf,fullfile(dirCache,[fileSave,'_xcorr.fig']));
print('-dpsc',pathSavePs,'-fillpage','-append');

% close;


%%% cross correlation between all pairs of occurrences for each sample
numRow = ceil(numSamples/2);
minOccurrences = min(occurrences);
maxOccurrences = max(occurrences);
vecT = 1:size(occurrences,1);
minXcraw = min(xcraw,[],[1,2]);
maxXcraw = max(xcraw,[],[1,2]);
for p=1:numComp
    minP = min(minOccurrences(:,:,p));
    maxP = max(maxOccurrences(:,:,p));
    for q=p+1:numComp
        figure('Colormap',jet(256),'PaperType','A4','PaperOrientation','portrait');
        minQ = min(minOccurrences(:,:,q));
        maxQ = max(maxOccurrences(:,:,q));
        minY = min(minP,minQ);
        maxY = max(maxP,maxQ);
        for r=1:numSamples
            
            subplot(numRow,8,r*4-4+(1:2));
            plot(vecT,occurrences(:,r,p),vecT,occurrences(:,r,q));
            grid on;
            xlim([0,max(vecT)]);
            ylim([minY,maxY]);
            title(sprintf("comp= #%d (blue) & #%d (red), sample= #%d",p,q,idxSamples(r)));
            %             if r==2
            %                 legend({['#',num2str(p)],['#',num2str(q)]});
            %             end
            if r<numSamples-1
                set(gca,'XTickLabel',[]);
            end
            
            subplot(numRow,8,r*4-1);
            plot(lags,xcraw(:,r,p,q),'-');
            grid on;
            ylim([minXcraw(1,1,p,q),maxXcraw(1,1,p,q)]);
            
            subplot(numRow,8,r*4);
            plot(occurrences(:,r,p),occurrences(:,r,q),'-');
            grid on;
            axis equal;
            xlim([minP,maxP]);
            ylim([minQ,maxQ]);
        end
        
        saveas(gcf,fullfile(dirCache,sprintf('%s_comp1=%d_comp2=%d.fig',fileSave,p,q)));
        print('-dpsc',pathSavePs,'-fillpage','-append');
        
        % close;
        
    end
end

fprintf('finished in %0.3f sec\n',toc(hTic));

end


function [occurrences,motifs,tcrsRec] = doTdeRica(tcrsNArranged,dimEmbed,numComp,subsample,dirCache)
%%% Reconstruction Independent Compnent Analysis (RICA) of embedded
%%% timecourses by Time Delay Embedding (TDE)

%%% Doing calculation for tde-rica will take too long time.
%%% Obtined results will be saved in the mat file in the cache directory.

h = tic();
fprintf('Do TDE-RICA ... ');

pathSaveTdeRica = fullfile(dirCache,...
    sprintf('tderica_basic_dimEmbed=%d_numComp=%d_subSample=%d.mat',...
    dimEmbed,numComp,subsample));

tcrsSubsampled = tcrsNArranged(1:subsample:end,:,:);
[maxT,numCells,numSamples] = size(tcrsSubsampled);

if ~exist(pathSaveTdeRica,'file')
    %%% TDE
    fprintf('calculating ... ');
    numTEmbed = maxT-dimEmbed+1;
    tmpTcrsEmbed = nan(numTEmbed,dimEmbed,numCells,numSamples);
    for p=1:numSamples
        tmpTcrsEmbed(:,:,:,p) = delayembed(tcrsSubsampled(:,:,p),dimEmbed);
    end
    tmpTcrsEmbed = reshape(permute(tmpTcrsEmbed,[1,4,2,3]),...
        [numTEmbed*numSamples,dimEmbed*numCells]);
    isTValid = ~any(isnan(tmpTcrsEmbed),2);
    tcrsEmbed = tmpTcrsEmbed(isTValid,:);
    
    %%% RICA, transposed, super-gaussian
    mdl = rica(tcrsEmbed',numComp,'IterationLimit',2000,...
        'NonGaussianityIndicator',ones(1,numComp));
    W = mdl.TransformWeights;
    M = transform(mdl,tcrsEmbed')';
    
    clear tmpTcrsEmbed tcrsEmbed
    save(pathSaveTdeRica,'W','M','isTValid');
    
else
    fprintf('load from cache ... ');
    load(pathSaveTdeRica,'W','M','isTValid');
end

[occurrences,motifs,tcrsRec] = parseTdeRica(W,M,isTValid,numSamples,dimEmbed);


fprintf('finished in %0.3f sec\n',toc(h));

end


function [occurrences,motifs,tcrsRec] = parseTdeRica(W,M,isTValid,numSamples,dimEmbed)
%%% obtain motifs, occurrences, and reconstructed time courses from the
%%% result of tde-rica

% h = tic();
% fprintf('Parse TDE-RICA results ... ');

numComp = size(W,2);

occurrences = nan(numel(isTValid),numComp);
occurrences(isTValid,:) = W;
occurrences = reshape(occurrences,[],numSamples,numComp);
motifs = reshape(M,numComp,dimEmbed,[]);

numTEmbed = size(occurrences,1);
numCells = size(motifs,3);
numT = numTEmbed + dimEmbed - 1;
tcrsRec = nan(numT,numCells,numSamples);
for p=1:numSamples
    tcrsRec(:,:,p) = delayembed_inv(reshape(permute(occurrences(:,p,:),[1,3,2])*M,...
        [numTEmbed,dimEmbed,numCells]));
end

% fprintf('finished in %0.3f sec\n',toc(h));

end


function [idxSamplesSelected,idxCellsSelected] ...
    = selectDataWithoutMissingValue(tcrsNArranged,numSamplesSelect)
%%% select samples and neurons without missing values by greedy algorithm

h = tic();
fprintf('Select data without missing value ... ');

isExistingValue = permute(~all(isnan(tcrsNArranged),1),[2,3,1]);
isExistingValueTmp = isExistingValue;
isSamplesSelected = false(1,size(isExistingValue,2));

for p=1:numSamplesSelect
    isExistingValueSelected = all(isExistingValue(:,isSamplesSelected),2);
    [~,tmpidx] = max(sum(isExistingValueSelected & isExistingValueTmp,1));
    isSamplesSelected(1,tmpidx) = true;
    isExistingValueTmp(:,tmpidx) = false;
end

idxSamplesSelected = find(isSamplesSelected);
idxCellsSelected = find(all(isExistingValue(:,idxSamplesSelected),2));

fprintf('finished in %0.3f sec\n',toc(h));

end


function [tcrsNArranged,strNamesUnique,posArranged,moveRaw] ...
    = loadData(dirRaw,strHeaderRemove,frameRemove,idxNormalize,dirCache)

%%% load raw data from excel files
%%% Reading raw data from excel files is too slow;
%%% Obtined results will be saved in the mat file in the cache directory.
h = tic();
fprintf('Load raw data ... ');

filesRaw = dir(fullfile(dirRaw,'*.xlsx'));
pathSaveRaw = fullfile(dirCache,'tderica_RawData.mat');
numSamples = numel(filesRaw);

if ~exist(pathSaveRaw,'file')
    % load data
    fprintf('from excel ... ')
    tcrsRaw     = cell(numSamples,1);
    strNamesRaw = cell(numSamples,1);
    posRaw      = cell(numSamples,1);
    moveRaw     = cell(numSamples,1);
    parfor p=1:numSamples
        tmpFile = fullfile(filesRaw(p).folder,filesRaw(p).name);
        [tcrsRaw{p},strNamesRaw{p},posRaw{p},moveRaw{p}] = loadRaw(tmpFile); %#ok<PFOUS>
        for q=1:numel(strHeaderRemove)
            if startsWith(filesRaw(p).name,strHeaderRemove{q})
                tcrsRaw{p} = tcrsRaw{p}(1:frameRemove(q),:,:); %#ok<PFBNS>
                posRaw{p}  =  posRaw{p}(1:frameRemove(q),:,:);
                moveRaw{p} = moveRaw{p}(1:frameRemove(q),:,:);
                break;
            end
        end
    end
    save(pathSaveRaw);
else
    fprintf('from cache ... ')
%     load(pathSaveRaw,'tcrsRaw','strNamesRaw');
    load(pathSaveRaw,'tcrsRaw','strNamesRaw','posRaw','moveRaw');
    numSamples = numel(tcrsRaw);
end


%%% filtering and normalizing
normalize = str2func(sprintf('normalize%d',idxNormalize));
tcrsN       = cell(numSamples,1);
flagOutlier = cell(numSamples,1);
strNames    = cell(numSamples,1);
posFiltered = cell(numSamples,1);
for p=1:numel(tcrsRaw)
    [tcrsN{p},flagOutlier{p}] = normalize(tcrsRaw{p}(:,:,2),tcrsRaw{p}(:,:,3));
    strNames{p} = strNamesRaw{p}(~flagOutlier{p});
    posFiltered{p} = posRaw{p}(:,~flagOutlier{p},:);
end


%%% arrange to 3-dimensional matrix
tmpStrNamesUnique = unique(cat(2,strNames{:})');
strNamesUnique = tmpStrNamesUnique(...
    isnan(cellfun(@str2double,tmpStrNamesUnique)) ...
    & ~startsWith(tmpStrNamesUnique,'HYPL')...
    & ~strcmpi(tmpStrNamesUnique,'FLP'));
strcmp2i = @(x,y)strcmpi(x(:,ones(1,numel(y))),y(ones(1,numel(x)),:));

maxT = max(cellfun(@size,tcrsN,num2cell(ones(size(tcrsN)))));
tcrsNArranged = nan(maxT,numel(strNamesUnique),numSamples);
posArranged   = nan(maxT,numel(strNamesUnique),numSamples);
for p=1:numSamples
    [idx1,idx2] = find(strcmp2i(strNamesUnique(:),strNames{p}(:)'));
    tcrsNArranged(1:size(tcrsN{p},1),idx1,p) = tcrsN{p}(:,idx2);
    posArranged(1:size(posFiltered{p},1),idx1,p) = posFiltered{p}(:,idx2);
end

fprintf('finished in %0.3f sec\n',toc(h));
end


function [tcrs,strNames,pos,move] = loadRaw(fileRaw)
[~,~,tmpMcherry] = xlsread(fileRaw,'pi_k');
[~,~,tmpCfp] = xlsread(fileRaw,'pi_k_Ch2');
[~,~,tmpYfp] = xlsread(fileRaw,'pi_k_Ch3');
[~,~,tmpXc] = xlsread(fileRaw,'xc');
[~,~,tmpYc] = xlsread(fileRaw,'yc');
[~,~,tmpZc] = xlsread(fileRaw,'zc');
[~,~,tmpMove] = xlsread(fileRaw,'Sheet1');
strNames = cellfun(@num2str,tmpMcherry(1,2:end),'UniformOutput',false);
tcrsMcherry = cell2mat(tmpMcherry(2:end,2:end));
tcrsCfp = cell2mat(tmpCfp(2:end,2:end));
tcrsYfp = cell2mat(tmpYfp(2:end,2:end));
xc = cell2mat(tmpXc(2:end,2:end));
yc = cell2mat(tmpYc(2:end,2:end));
zc = cell2mat(tmpZc(2:end,2:end));
tcrs = cat(3,tcrsMcherry,tcrsCfp,tcrsYfp);
pos = cat(3,xc,yc,zc);
move = cell2mat(tmpMove(2:end,2:end));
end


function [tcrsFN,flagOutlier] = normalize1(tcrsCfp,tcrsYfp)
tcrs = medfilt1(tcrsYfp./tcrsCfp,11);

%%% denoising
flagOutlier = any(isnan(tcrs) | tcrs<0 | tcrs>10);
tcrs = tcrs(:,~flagOutlier);

%%% filtering
tcrsFilt = sgolayfilt(tcrs,3,101);

%%% normalization
% tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,min(tcrsFilt)),max(tcrsFilt)-min(tcrsFilt)); % min=0,max=1
tcrsFN = bsxfun(@rdivide,tcrsFilt,mean(tcrsFilt))-1; % R / Rmean - 1
% tcrsFN = tcrsFilt; % no normalization
end


function [tcrsFN,flagOutlier] = normalize2(tcrsCfp,tcrsYfp)

%%% denoising by median filter
tcrsCfpMed = medfilt1(tcrsCfp,5,'omitnan','truncate');
tcrsYfpMed = medfilt1(tcrsYfp,5,'omitnan','truncate');


%%% remove outlier
tmpMedian = median(tcrsCfpMed);
flagUnderMedian = (tcrsCfpMed < tmpMedian/10);
tcrsCfpMed2 = tcrsCfpMed;
tcrsYfpMed2 = tcrsYfpMed;
tcrsCfpMed2(flagUnderMedian) = nan;
tcrsYfpMed2(flagUnderMedian) = nan;
tcrsCfpMed3 = medfilt1(tcrsCfpMed2,4,'omitnan','truncate');
tcrsYfpMed3 = medfilt1(tcrsYfpMed2,4,'omitnan','truncate');
tcrsCfpMed4 = tcrsCfpMed2;
tcrsYfpMed4 = tcrsYfpMed2;
tcrsCfpMed4(flagUnderMedian) = tcrsCfpMed3(flagUnderMedian);
tcrsYfpMed4(flagUnderMedian) = tcrsYfpMed3(flagUnderMedian);

flagOutlier = sum(flagUnderMedian)>400 | any(isnan(tcrsCfpMed4)) | any(isnan(tcrsYfpMed4));
tcrs = tcrsYfpMed4./tcrsCfpMed4;
tcrs = tcrs(:,~flagOutlier);

%%% detrending by linear fitting
tvec = (1:size(tcrs,1))';
coeff = [ones(size(tvec)),tvec]\tcrs;
tcrsFilt = bsxfun(@minus,tcrs,coeff(1,:))-tvec*coeff(2,:);

% %%% filtering
% tcrsFilt = sgolayfilt(tcrs,3,31);

%%% normalization
% tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,min(tcrsFilt)),max(tcrsFilt)-min(tcrsFilt)); % min=0,max=1
tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,mean(tcrsFilt)),std(tcrsFilt)); % mean=0,sd=1
% tcrsFN = bsxfun(@rdivide,tcrsFilt,mean(tcrsFilt))-1; % R / Rmean - 1
% tcrsFN = tcrsFilt; % no normalization
end


function [tcrsFN,flagOutlier] = normalize3(tcrsCfp,tcrsYfp)

%%% denoising by median filter
tcrsCfpMed = medfilt1(tcrsCfp,5,'omitnan','truncate');
tcrsYfpMed = medfilt1(tcrsYfp,5,'omitnan','truncate');


%%% remove outlier
tmpMedian = median(tcrsCfpMed);
flagUnderMedian = (tcrsCfpMed < tmpMedian/10);
tcrsCfpMed2 = tcrsCfpMed;
tcrsYfpMed2 = tcrsYfpMed;
tcrsCfpMed2(flagUnderMedian) = nan;
tcrsYfpMed2(flagUnderMedian) = nan;
tcrsCfpMed3 = medfilt1(tcrsCfpMed2,4,'omitnan','truncate');
tcrsYfpMed3 = medfilt1(tcrsYfpMed2,4,'omitnan','truncate');
tcrsCfpMed4 = tcrsCfpMed2;
tcrsYfpMed4 = tcrsYfpMed2;
tcrsCfpMed4(flagUnderMedian) = tcrsCfpMed3(flagUnderMedian);
tcrsYfpMed4(flagUnderMedian) = tcrsYfpMed3(flagUnderMedian);

flagOutlier = sum(flagUnderMedian)>400 | any(isnan(tcrsCfpMed4)) | any(isnan(tcrsYfpMed4));
tcrs = tcrsYfpMed4./tcrsCfpMed4;
tcrs = tcrs(:,~flagOutlier);

%%% detrending by linear fitting
tvec = (1:size(tcrs,1))';
coeff = [ones(size(tvec)),tvec]\tcrs;
tcrsDetrend = bsxfun(@minus,tcrs,coeff(1,:))-tvec*coeff(2,:);


%%% filtering
tcrsFilt = sgolayfilt(tcrsDetrend,3,31);

%%% normalization
% tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,min(tcrsFilt)),max(tcrsFilt)-min(tcrsFilt)); % min=0,max=1
tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,mean(tcrsFilt)),std(tcrsFilt)); % mean=0,sd=1
% tcrsFN = bsxfun(@rdivide,tcrsFilt,mean(tcrsFilt))-1; % R / Rmean - 1
% tcrsFN = tcrsFilt; % no normalization
end


function [tcrsFN,flagOutlier] = normalize4(tcrsCfp,tcrsYfp)

%%% denoising by median filter
tcrsCfpMed = medfilt1(tcrsCfp,5,'omitnan','truncate');
tcrsYfpMed = medfilt1(tcrsYfp,5,'omitnan','truncate');


%%% remove outlier
tmpMedian = median(tcrsCfpMed);
flagUnderMedian = (tcrsCfpMed < tmpMedian/10);
tcrsCfpMed2 = tcrsCfpMed;
tcrsYfpMed2 = tcrsYfpMed;
tcrsCfpMed2(flagUnderMedian) = nan;
tcrsYfpMed2(flagUnderMedian) = nan;
tcrsCfpMed3 = medfilt1(tcrsCfpMed2,4,'omitnan','truncate');
tcrsYfpMed3 = medfilt1(tcrsYfpMed2,4,'omitnan','truncate');
tcrsCfpMed4 = tcrsCfpMed2;
tcrsYfpMed4 = tcrsYfpMed2;
tcrsCfpMed4(flagUnderMedian) = tcrsCfpMed3(flagUnderMedian);
tcrsYfpMed4(flagUnderMedian) = tcrsYfpMed3(flagUnderMedian);

flagOutlier = sum(flagUnderMedian)>400 | any(isnan(tcrsCfpMed4)) | any(isnan(tcrsYfpMed4));
tcrs = tcrsYfpMed4./tcrsCfpMed4;
tcrs = tcrs(:,~flagOutlier);

%%% detrending by linear fitting
tvec = (1:size(tcrs,1))';
coeff = [ones(size(tvec)),tvec]\tcrs;
tcrsFilt = bsxfun(@minus,tcrs,coeff(1,:))-tvec*coeff(2,:);

% %%% filtering
% tcrsFilt = sgolayfilt(tcrs,3,31);

%%% normalization
% tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,min(tcrsFilt)),max(tcrsFilt)-min(tcrsFilt)); % min=0,max=1
% tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,mean(tcrsFilt)),std(tcrsFilt)); % mean=0,sd=1
tcrsFN = bsxfun(@rdivide,bsxfun(@minus,tcrsFilt,mean(tcrsFilt)),std(diff(tcrsFilt))); % mean=0,sd(diff)=1
% tcrsFN = bsxfun(@rdivide,tcrsFilt,mean(tcrsFilt))-1; % R / Rmean - 1
% tcrsFN = tcrsFilt; % no normalization
end


function tcrsEmbed = delayembed(tcrs,dimEmbed)
tcrsEmbed = zeros(size(tcrs,1)-dimEmbed+1,dimEmbed,size(tcrs,2));
for p=1:dimEmbed
    tcrsEmbed(:,p,:) = tcrs(p:end-dimEmbed+p,:);
end
end


function tcrsRaw = delayembed_inv(tcrsEmbed)
[numTEmbed,dimEmbed,numUseCell] = size(tcrsEmbed);
numT = numTEmbed+dimEmbed-1;
tmpTcrs = nan(numT,dimEmbed,numUseCell);
for p=1:dimEmbed
    tmpTcrs(p:end-dimEmbed+p,p,:) = tcrsEmbed(:,p,:);
end
tcrsRaw = permute(mean(tmpTcrs,2,'omitnan'),[1,3,2]);
end


