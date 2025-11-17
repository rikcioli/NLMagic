function skew(X::Matrix{T}) where {T}
    return (X - X')/2
end

function skew(arrX::Vector{<:Matrix})
    return map(skew, arrX)
end

function project(U::Matrix{T}, D::Matrix{T}) where {T}
    return U * skew(U' * D)
end

function project(arrU::Vector{<:Matrix}, arrD::Vector{<:Matrix})
    return arrU .* skew([U' for U in arrU] .* arrD)
end

function extractU(M::Matrix{T}) where {T}
    U, S, V = svd(M)
    W = U*V'
    return W
end

function extractU(arrM::Vector{<:Matrix})
    return map(extractU, arrM)
end


#move  U in the direction of X with step length t, 
#X is the gradient obtained using projection.
#return both the "retracted" unitary as well as the tangent vector at the retracted point
# always stabilize unitarity and skewness, it could leave tangent space due to numerical errors
function retract(U::Matrix{T}, X::Matrix{T}, t::Float64) where {T}

    # ensure unitarity of U
    U_unitary = extractU(U)

    #non_unitarity = norm(U - U_polar)/length(U)
    #if non_unitarity > 1e-10
    #    @show non_unitarity
    #end
    U = U_unitary

    Uinv = U'
    X_id = Uinv * X

    # ensure skewness of X_id
    #non_skewness = norm(X_id - skew(X_id))/length(U)
    #if non_skewness > 1E-10
    #    @show non_skewness
    #end
    # if non_skewness > 1E-15 + eps()
    #     throw(DomainError(non_skewness, "X is not in the tangent space at U"))
    # end
    X_id = skew(X_id)

    # construct the geodesic at the tangent space at unity
    # then move it to the correct point by multiplying by U
    U_new = U * exp(t*X_id)

    # move X to the new tangent space U_new
    X_new = U_new * X_id #move first to the tangent space at unity, then to the new point
    return U_new, X_new
end

function retract(arrU::Vector{<:Matrix}, arrX::Vector{<:Matrix}, t::Float64)
    
    arrU_unitary = extractU(arrU)
    arrU = arrU_unitary
    arrUinv = [U' for U in arrU]

    arrX_id = arrUinv .* arrX
    arrX_id = skew(arrX_id)

    arrU_new = arrU .* map(X -> exp(t*X), arrX_id)
    arrX_new = arrU_new .* arrX_id #move first to the tangent space at unity, then to the new point
    return arrU_new, arrX_new
end

function inner(U::Matrix{T}, X::Matrix{T}, Y::Matrix{T}) where {T}
    return real(tr(X'*Y))
end

function inner(X::Matrix{T}, Y::Matrix{T}) where {T}
    return real(tr(X'*Y))
end

function inner(arrU::Vector{<:Matrix}, arrX::Vector{<:Matrix}, arrY::Vector{<:Matrix})
    innerprod = 0.
    for i in eachindex(arrX)
        innerprod += real(tr(arrX[i]'*arrY[i]))
    end
    return innerprod
    #return real(tr(arrX'*arrY))
end
function inner(arrX::Vector{<:Matrix}, arrY::Vector{<:Matrix})
    innerprod = 0.
    for i in eachindex(arrX)
        innerprod += real(tr(arrX[i]'*arrY[i]))
    end
    return innerprod
    #return real(tr(arrX'*arrY))
end

#parallel transport
"""transport tangent vector ξ along the retraction of x in the direction η (same type as a gradient) 
with step length α, can be in place but the return value is used. 
Transport also receives x′ = retract(x, η, α)[1] as final argument, 
which has been computed before and can contain useful data that does not need to be recomputed"""
function transport!(ξ, U::Matrix{T}, η, α, U_new::Matrix{T}) where {T}
    Uinv = U'
    ξ = U_new * Uinv * ξ
    return ξ
end

function transport!(ξ, arrU::Vector{<:Matrix}, η, α, arrU_new::Vector{<:Matrix})
    arrUinv = [U' for U in arrU]
    ξ = arrU_new .* arrUinv .* ξ
    return ξ
end