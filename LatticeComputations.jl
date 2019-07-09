using Nemo
import AbstractAlgebra
import IterTools.subsets



K, t = PuiseuxSeriesField(QQ, 100, "t")




function indep_lattice(A)

	d = size(A)[1]

	indep_A = Array{Rational{Int64}}(undef,d,1)
	indep_A[d] = valuation(A[d,d])

	for i = 1:d-1
		k = d-i

		container = copy(A[1:d,k])

		for j = k+1:d
			val = valuation(A[j,j]) - valuation(container[j])

			if val >= 0

				container = t^val .* container - (t^val * container[j]//A[j,j]) .* A[1:d,j]

			end
		end

		indep_A[k] = valuation(container[k])

	end

	return indep_A

end


function normal_form(B)

	n,d = size(B)

	A = copy(B)

	switch = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}(undef,n,1)


	for i = 1:n

		j = argmin(valuation.(A[i,i:d])) + i - 1

		switch = A[1:n,i]


		A[1:n,i] = A[1:n,j]
		A[1:n,j] = switch

		val = valuation(A[i,i])
		A[1:n,i] = t^(val) .* (A[1:n,i] .// A[i,i])

		for k = i+1:d
			alpha = - A[i,k]//A[i,i]
			A[1:n,k] = A[1:n,k] + alpha .* A[1:n,i]
		end

	end

	return A[1:n,1:n]

end


function compute_Polynomial(A)

	d = size(A)[1]

	coefs = Dict{Array{Int64},Int64}([]=>0)

	for I in IterTools.subsets(1:d,1)
		B = normal_form(A[I,1:d])
		coefs[I] = valuation(B[1,1])
	end

	for I in IterTools.subsets(1:d,2)
		B = normal_form(A[I,1:d])
		coefs[I] = valuation(B[1,1]) + valuation(B[2,2])
	end

	if d >= 3

		for k = 3:d

			for I in subsets(1:d,k)

				J = I[2:k]

				B = normal_form(A[I,1:d])
				L = indep_lattice(B)

				coefs[I] = coefs[J] + L[1]

			end

		end


	end

	return coefs

end




A = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}([t^0    0t     0t     0t     0t   0t ;
								 t^0    t      0t     0t     0t   0t ;
							 	 t^0    t^2    t^3    0t     0t   0t ;
							 	 t^0    t^3    t^2    t^4    0t   0t ;
								 t^0    t^2    t^2    t      t^3   0t ;
								 t^0    t      t^2    t^3     t^4   t^5 ])




d = size(A)[1]

P = compute_Polynomial(A)

println(P)
