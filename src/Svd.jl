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
    dₙ = size(S, 1)

    return U, Σ, Vᵀ, dₙ

end