import math

def karatsuba(x,y):
	#recursive base case, halts the function
	if (len(str(x)) <= 5 or len(str(y)) <= 5):
		product = x*y

	else:
		#need to determine which is the larger number
		#and assign values to placeholders accordingly
		if (len(str(x)) >= len(str(y))):
			alphaSize_n = len(str(x))
			alpha = x
			betaSize_m = len(str(y))
			beta = y
		else:
			betaSize_m = len(str(x))
			beta = x
			alphaSize_n = len(str(y))
			alpha = y

		#need to half the exponent 
		n_Over2 = int(math.floor(alphaSize_n / 2))
		m_Over2 = int(math.floor(betaSize_m / 2))

		#need to get the values for placeholders in algorithm
		alpha_1 = int(math.floor(alpha / (10 ** n_Over2))) 
		alpha_0 = int(alpha - (alpha_1 * (10 ** n_Over2)))

		beta_1 = int(math.floor(beta / (10 ** m_Over2)))
		beta_0 = int(beta - (beta_1 * (10 ** m_Over2)))

		z_1 = karatsuba(alpha_1 , beta_1)
		z_3 = karatsuba(alpha_0 , beta_0)

		if (alphaSize_n == betaSize_m):
			z_2 = karatsuba((alpha_1 + alpha_0) , (beta_1 + beta_0)) - z_1 - z_3
			product = ((10 ** alphaSize_n) * z_1) + ((10 ** n_Over2) * z_2) + z_3  

		else:
			z_2 = karatsuba(alpha_0 , beta_1) + (10 ** (n_Over2 - m_Over2)) * karatsuba(alpha_1 , beta_0)
			product = ((10 ** (n_Over2 + m_Over2)) * z_1) + ((10 ** m_Over2) * z_2) + z_3

	return(product)

print (karatsuba(12223334444,12223334444))