module selector_abstract_types
using Test, RandLinearAlgebra
import RandLinearAlgebra: complete_selector, update_selector!, select_indices!


#############################
# Initial Testing Parameters
#############################
struct TestSelector <: Selector end
mutable struct TestSelectorRecipe <: SelectorRecipe
    code::Int64
end

########################################
# Tests for Selector Abstract Types
########################################
@testset "Selector Abstract Types" begin
    @test isdefined(Main, :Selector)
    @test isdefined(Main, :SelectorRecipe)
end

@testset "Selector Completion Errors" begin
    A = ones(2,2)
        
    @test_throws ArgumentError complete_selector(TestSelector(), A)
end

#############################
# Update Testing Parameters 
#############################
complete_selector(selector::TestSelector, A::AbstractMatrix) = 
    TestSelectorRecipe(1)

@testset "Selector Completion" begin
    A = ones(2, 2)

    select = complete_selector(TestSelector(), A) 
    @test select isa TestSelectorRecipe
    @test select.code == 1
end

###############################
# Test Updating selector errors
###############################

@testset "Selector Update Errors" begin
    A = ones(2,2)
        
    select = complete_selector(TestSelector(), A) 
    @test_throws ArgumentError update_selector!(select, A)
    @test_throws ArgumentError update_selector!(select)
    
end

###############################
# Test update_selector
###############################


@testset "Selector Updating" begin
    A = ones(2, 2)
    function update_selector!(selector::TestSelectorRecipe, A::AbstractMatrix)
        selector.code = 2
    end

    function update_selector!(selector::TestSelectorRecipe)
        selector.code = 2
    end

    # test 2 arg selector
    let A = A
        select = complete_selector(TestSelector(), A) 
        update_selector!(select, A)
        @test select isa TestSelectorRecipe
        @test select.code == 2
    end

    # test 1 arg selector
    let A = A
        select = complete_selector(TestSelector(), A) 
        update_selector!(select)
        @test select isa TestSelectorRecipe
        @test select.code == 2
    end

end

#################################
# Test selection errors
#################################
@testset "Select Indices" begin
    A = ones(2,2)
    select = complete_selector(TestSelector(), A)
    idx = zeros(Int64, 2)
    n_idx = 2
    start_idx = 1
    
    @test_throws ArgumentError select_indices!(idx, select, A, n_idx, start_idx)
end

end
