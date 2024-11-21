classdef SITFO < ALGORITHM
% <single> <real> <large/none> <expensive>
% Surrogate information transfer and fusion optimization algorithm
% ThetaStar ---  1e-5 --- pre-specified threshold 
% maxiter --- 10 --- maximum number of iterations
% nMCpoints --- 10000 --- number of MC samples

%------------------------------- Reference --------------------------------
% Pang, Y., Zhang, S., Jin, Y., Wang, Y., Lai, X., & Song, X. (2024). 
% Surrogate information transfer and fusion in high-dimensional expensive 
% optimization problems. Swarm and Evolutionary Computation, 88, 101586.

    methods
        function main(Algorithm,Problem)
         
            %% Parameter setting
            [ThetaStar,maxiter,  nMCpoints] = Algorithm.ParameterSet(1e-5,10,10000);
            
            %% Generate the random population
            N          = Problem.N;
            P          = UniformPoint(N,Problem.D,'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper-Problem.lower,N,1).*P+repmat(Problem.lower,N,1));
            MaxNode    = 20;
            BU         = Problem.upper;
            BD         = Problem.lower;
            THETA=5;


          %% Initialize swarm
            Swarm = [Population.decs,Population.objs];
            Kriging_objs = Population.objs;
            Kriging_mse = ones(size(Kriging_objs));
            Kriging_dmse = zeros(N,Problem.D);
            DeltaRBF = repmat(BU-BD,N,1).*rand(N,1)+repmat(BD,N,1);
            weight=5;
            Positive_mse = Kriging_mse;
            
          %% Optimization
            
            A   = Population;
            while Algorithm.NotTerminated(A)              
                spread  =  sqrt(sum((max(A.decs,[],1)-min(A.decs,[],1)).^2));
                train_X = A.decs;
                train_Y = A.objs;
                [~,distinct] = unique(round(train_X*1e6)/1e6,'rows');  
                train_X   = train_X(distinct,:);
                train_Y   = train_Y(distinct,:);
                 
                % information transfer
                if  std(Positive_mse)/mean(Positive_mse)<ThetaStar
                    rbfnet = srgtsnewrb(train_X',train_Y',0.1,spread,MaxNode,1,'off');
                    output = MCGlobalSensitivity(rbfnet,[Problem.lower;Problem.upper],nMCpoints);
                    sensitivity=output.individual.';
                    weight=sensitivity*Problem.D/sum(sensitivity);
                    KrigingModel   = dacefit_tr(train_X,train_Y,'regpoly0','corrgauss',THETA, weight, [1e-5], [100]);
                    THETA = KrigingModel.theta;
                else
                    rbfnet = srgtsnewrb(train_X',train_Y',0.1,spread,MaxNode,1,'off');
                    KrigingModel   = dacefit_tr(train_X,train_Y,'regpoly0','corrgauss',THETA, weight);
                    THETA = KrigingModel.theta;
                end

                
                for j =1:maxiter
                    Demons = Swarm;
                    if  std(Positive_mse)/mean(Positive_mse)>ThetaStar
                        [Swarm,DeltaRBF] = SLPSOMSE(rbfnet,Demons,Swarm,DeltaRBF,Problem,Kriging_dmse);
                    else
                        [Swarm,DeltaRBF] = SLPSO(rbfnet,Demons,Swarm,DeltaRBF,Problem);
                    end 
                    
                    range = mean(Problem.upper - Problem.lower);
                    for i = 1:N
                        [Kriging_objs(i),~,Kriging_mse(i),Kriging_dmse(i,:)] = predictor(Swarm(i,1:Problem.D),KrigingModel);
                        Kriging_mse(i) = max(Kriging_mse(i),0);
                        rho = (1/norm(Kriging_dmse(i,:))).*0.1.*range.*exp(-(Problem.FE-Problem.N)) ;
                        Kriging_dmse(i,:) = Kriging_dmse(i,:).* rho;
                    end                     

                    Positive_mse = Kriging_mse;
                    Positive_mse(find(Positive_mse<1e-5),:) = [];
                end
                             
                

                
              %%  Infill
                if std(Positive_mse)/mean(Positive_mse)>ThetaStar
                    %add point considering uncertainty
                    [value,bestI] = sort(Swarm(:,1+Problem.D));
                    new1=[];            
                    for i = 1:N
                        dec=Swarm(bestI(i),1:Problem.D);
                        index=ismember(dec,A.decs,'rows');
                        if index==0
                            new1 = Problem.Evaluation(dec);  
                            Swarm(bestI(i),Problem.D+1) = new1.objs;                     
                            break;
                        end 
                     end
                    if ~isempty(new1)
                        A = [A,new1];
                    end
                
          
                     loc = OptEI(A,KrigingModel,Problem,Swarm(randi(N),1:Problem.D));
                     new2 = Problem.Evaluation(loc);  
 
                     if ~isempty(new2)
                        A = [A,new2];
                     end   
                else
                     %add point without considering uncertainty
                    [value,bestI] = sort(Swarm(:,1+Problem.D));
                    new1=[];            
                    for i = 1:N
                        dec=Swarm(bestI(i),1:Problem.D);
                        index=ismember(dec,A.decs,'rows');
                        if index==0
                            new1 = Problem.Evaluation(dec);  
                            Swarm(bestI(i),Problem.D+1) = new1.objs;                  
                            break;
                        end 
                     end
                    if ~isempty(new1)
                    A = [A,new1];
                    end
                end    
            end  
        end
    end
end