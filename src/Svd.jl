using LinearAlgebra

function decompose(w)
    d₁ = size(w, 1)
    d₂ = size(w, 2)

    F = svd(w, full=true)
    U = F.U
    S = F.S
    # S[abs.(S) .< 0.00001] .= 0
    # filter!(x->x ≠ 0,S)
    # println(size(S))
    # println(S)

    Vᵀ = F.Vt

    dₙ = size(S, 1)
    Σ = zeros(d₁, d₂)
    for i in eachindex(S)
        Σ[i, i] = S[i]
    end

    Σ[abs.(Σ) .< 0.000000001] .= 0

    Pₗ = zeros(d₁, d₁)
    for i = reverse(1:d₁)
        for j in 1:d₁
            if(i+j == d₁+1)
                Pₗ[j, i] = 1
            end
        end
    end
    
    Pᵣ = zeros(d₂, d₂)
    for i = reverse(1:d₁)
        for j in 1:d₁
            if (i+j == d₁+1)
                Pᵣ[j, i] = 1
            end
        end
    end
    for i in reverse(d₁+1:d₂)
        Pᵣ[i, i] = 1
    end

    Uᵀ = transpose(U)
    V = transpose(Vᵀ)

    Uᵀₚ = Pₗ * Uᵀ
    Vₚ = V * Pᵣ

    Uₚ = transpose(Uᵀₚ)
    Vᵀₚ = transpose(Vₚ)

    Σₚ = Pₗ * Σ * Pᵣ

    return Uₚ, Σₚ, Vᵀₚ, dₙ
end

function permute_variables(U)
    d = size(U, 1)
    Pᵣ = zeros(d, d)
    for i = reverse(1:d)
        for j in 1:d
            if(i+j == d+1)
                Pᵣ[j, i] = 1
            end
        end
    end
    return U * Pᵣ
end