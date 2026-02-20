module basic_logger
    using Test, RandLinearAlgebra, Random
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")
    using .FieldTest
    using .ApproxTol
    @testset "Logger BasicLogger" begin
        Random.seed!(21321)
        n_rows = 4
        n_cols = 2
    	A = rand(n_rows, n_cols)
    	b = rand(n_rows)
        
        @testset "Basic Logger" begin
            @test supertype(BasicLogger) == Logger
            # test field names and types
            @test fieldnames(BasicLogger) == (
                :max_it, :collection_rate, :threshold_info, :stopping_criterion
            )
            @test fieldtypes(BasicLogger) == (
                Int64, Int64, Union{Float64, Tuple}, Function
            )

            # test internal structure
            let 
                max_it = -1
                cr = 1
                threshold = 1e-2
                stop = threshold_stop

                @test_throws ArgumentError(
                    "Field `max_it` must be positive or 0."
                ) BasicLogger(max_it, cr, threshold, stop)
            end

            let 
                max_it = 0
                cr = 0
                threshold = 1e-2
                stop = threshold_stop

                @test_throws ArgumentError(
                    "Field `colection_rate` must be positive."
                ) BasicLogger(max_it, cr, threshold, stop)
            end

            let 
                max_it = 1
                cr = 2
                threshold = 1e-2
                stop = threshold_stop

                @test_throws ArgumentError(
                    "Field `colection_rate` must be smaller than `max_it`."
                ) BasicLogger(max_it, cr, threshold, stop)
            end

            # Test key word defaults in constructor
            let 
        	    L1 = BasicLogger()
    
        	    @test typeof(L1) <: Logger 
        	    @test L1.max_it == 0
        	    @test L1.collection_rate == 1
                @test L1.threshold_info == 0
        	    @test L1.stopping_criterion == threshold_stop
            end
        end

        @testset "Basic Logger Recipe" begin
            
            supertype(BasicLoggerRecipe) == LoggerRecipe
            # test that the recipe aligns with the common requirements
            @test_logger BasicLoggerRecipe
            # Test fieldnames and types
            @test fieldnames(BasicLoggerRecipe) == (
                :max_it,
                :error, 
                :threshold_info, 
                :iteration, 
                :record_location, 
                :collection_rate, 
                :converged, 
                :stopping_criterion, 
                :hist
            )
            @test fieldtypes(BasicLoggerRecipe) == (
                Int64, 
                Float64, 
                Union{Float64, Tuple}, 
                Int64, 
                Int64, 
                Int64, 
                Bool, 
                Function,
                Vector{Float64}
            )
        end

        @testset "Basic Logger: Complete Compressor" begin
            # Test default initlialization
            let A = deepcopy(A),
                b = deepcopy(b),
                L1 = BasicLogger(max_it = 3 * n_rows),
        	    # Test completion of constructor
        	    L_method = complete_logger(L1)

        	    @test typeof(L_method) <: LoggerRecipe
                # test the initialization of logger
                @test L_method.max_it == 3 * n_rows
                @test size(L_method.hist, 1) == 3 * n_rows + 1 
                @test L_method.stopping_criterion == threshold_stop
                @test L_method.threshold_info == 0
                @test L_method.converged == false
                @test L_method.iteration == 1 
                @test L_method.record_location == 1
            end


            # Test initialization with keyword inputs
            let max_it = 1000,
                cr = 2,
                threshold = 1e-6,
        	    L1 = BasicLogger(max_it = max_it,
                                 collection_rate = cr,
                                 threshold = threshold
                                ),
                A = deepcopy(A),
                b = deepcopy(b)
    
        	    @test typeof(L1) <: Logger 
        	    @test L1.max_it == max_it 
        	    @test L1.collection_rate == cr
                @test L1.threshold_info == threshold
        	    @test L1.stopping_criterion == threshold_stop
    
        	    # Test completion of constructor
        	    L_method = complete_logger(L1)
        	    @test typeof(L_method) <: LoggerRecipe
                # test the initialization of logger
                @test L_method.max_it == max_it 
                @test size(L_method.hist, 1) == Int(ceil(max_it / cr)) + 1
                @test L_method.stopping_criterion == threshold_stop
                @test L_method.threshold_info == threshold 
                @test L_method.converged == false
                @test L_method.iteration == 1 
                @test L_method.record_location == 1
            end

        end
        
        @testset "Basic Logger: Update Logger" begin
            # Test functionality of logger
            let A = deepcopy(A),
                b = deepcopy(b),
                L1 = BasicLogger(collection_rate = 2, threshold = 1e-5, max_it = 12),
        	    L_method = complete_logger(L1)
                # first check that nothing gets recorded at it 1
                update_logger!(L_method, 1e-2, 1)
                @test L_method.hist[1] == 0.0 
                @test L_method.record_location == 1
                @test L_method.iteration == 1
                @test L_method.converged == false
                # Test if the threshold is satisfied on first it a value is recorded
                update_logger!(L_method, 1e-6, 1)
                @test L_method.hist[1] == 1e-6 
                @test L_method.record_location == 1
                @test L_method.iteration == 1
                @test L_method.converged == true 
            end

            let A = deepcopy(A),
                b = deepcopy(b),
                L1 = BasicLogger(collection_rate = 2, threshold = 1e-5, max_it = 12),
        	    L_method = complete_logger(L1)
                # Check that a value is recorded at iteration 2
                update_logger!(L_method, 1e-2, 2)
                @test L_method.hist[1] == 1e-2 
                @test L_method.record_location == 2
                @test L_method.iteration == 2
                @test L_method.converged == false 
            end

            let A = deepcopy(A),
                b = deepcopy(b),
                L1 = BasicLogger(collection_rate = 2, threshold = 1e-5, max_it = 12),
        	    L_method = complete_logger(L1)
                # check that you stop when the max_it is satisfied max_it is 
                # 3 * num rows in A 
                update_logger!(L_method, 1e-1, 13)
                @test L_method.hist[1] == 1e-1 
                @test L_method.record_location == 1
                @test L_method.iteration == 13 
                @test L_method.converged == true 
                
        	end

        end

        @testset "Basic Logger: Reset Logger" begin
            # Check that the logger is reset
            let A = deepcopy(A),
                b = deepcopy(b),
                L1 = BasicLogger(collection_rate = 2, threshold = 1e-5, max_it = 12),
        	    L_method = complete_logger(L1)

                L_method.hist[2] = 23.0
                L_method.iteration = 2
                L_method.converged = true
                L_method.error = 23.0
                reset_logger!(L_method)
                @test L_method.error == 0.0 
                @test L_method.iteration == 1 
                @test L_method.record_location == 1 
                @test L_method.converged == false 
                @test sum(L_method.hist .!= 0.0) == 0
            end

        end

        @testset "Basic Logger: Threshold Stop" begin
            # Test the threshold stop function
            let A = deepcopy(A),
                b = deepcopy(b),
                L1 = BasicLogger(collection_rate = 2, threshold = 1e-5, max_it = 12),
        	    L_method = complete_logger(L1)

                L_method.error = 1.01e-5
                @test threshold_stop(L_method) == false
                L_method.error = 9.99e-6
                @test threshold_stop(L_method) == true 
            end

        end
    
    end

end
