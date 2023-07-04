using LinearAlgebra

function decompose(w)
    d₁ = size(w, 1)
    d₂ = size(w, 2)

    F = svd(w, full=true)
    U = F.U
    S = F.S
    Vᵀ = F.Vt

    S[abs.(S) .< 0.00001] .= 0
    filter!(x->x≠0.0,S)

    Σ = zeros(d₁, d₂)
    for i in eachindex(S)
        Σ[i, i] = S[i]
    end
    dₙ = d₂ #size(S, 1)

    return U, Σ, Vᵀ, dₙ

end

function lu_permute(Vᵀ)
    F = lu(Vᵀ, NoPivot())
    L = F.L
    U = F.U

    #Uₚ = permute_rows(U)
    #Uₚ = permute_columns(Uₚ)

    #Lₚ = permute_columns(L)
    
    return L, U
end

#=    Pₗ = zeros(d₁, d₁)
    for i = reverse(1:d₁)
        for j in 1:d₁
            if(i+j == d₁+1)
                Pₗ[j, i] = 1
            end
        end
    end
    
    Pᵣ = zeros(d₂, d₂)
    for i = reverse(1:d₂)
        for j in 1:d₂
            if (i+j == d₂+1)
                Pᵣ[j, i] = 1
            end
        end
    end
    #for i in reverse(d₁+1:d₂)
    #   Pᵣ[i, i] = 1
    #end

    Uᵀ = transpose(U)
    V = transpose(Vᵀ)

    Uᵀₚ = Pₗ * Uᵀ
    Vₚ = V * Pᵣ

    Uₚ = transpose(Uᵀₚ)
    Vᵀₚ = transpose(Vₚ)

    Σₚ = Pₗ * Σ * Pᵣ

    Vᵀₚ = permute_columns(Vᵀₚ) # only for the second approach

    return Uₚ, Σₚ, Vᵀₚ, dₙ
end

function permute_rows(M)
    d = size(M, 1)
    P = zeros(d, d)
    for i = reverse(1:d)
        for j in 1:d
            if(i+j == d+1)
                P[j, i] = 1
            end
        end
    end
    return P * M
end

function permute_columns(M)
    d = size(M, 1)
    P = zeros(d, d)
    for i = reverse(1:d)
        for j in 1:d
            if(i+j == d+1)
                P[j, i] = 1
            end
        end
    end
    return M * P
end

=#