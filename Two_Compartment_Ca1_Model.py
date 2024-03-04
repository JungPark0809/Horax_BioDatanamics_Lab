#two compartment ca1 cell from Lowet et el. 2016
# function two_comp_ca1(peram, T1,T2) takes input of parameters, time and input voltage
# This version returns membrane potential and all other gating variables



import numpy as np
import numba
import math as m


#Kinetic Equations
@numba.jit(nopython=True)
def d_h(V,h)-> float:
    return (((1.0/(1.0+np.exp(-(V+45.0)/(-7) )  )) ) - h)/(0.1+(0.75/(1+np.exp(-(V+40.5)/(-6.0)   ))) )

@numba.jit(nopython=True)
def d_n(V,n)-> float:
    return (((1.0/(1.0+np.exp(-(V+35.0)/(10) )  )) ) - n)/(0.1+(0.5/(1+np.exp(-(V+27.0)/(-15.0)   ))))

@numba.jit(nopython=True)
def d_b(V,b)-> float:
    return (((1.0/(1.0+np.exp(-(V+80.0)/(-6) )  )) ) - b)/(15.0)

@numba.jit(nopython=True)
def d_z(V,z)-> float:
    return (((1.0/(1.0+np.exp(-(V+39.0)/(5) )  )) ) - z)/(75.0)


@numba.jit(nopython=True)
def d_r(V,r):
    return (((1.0/(1.0+np.exp(-(V+20.0)/(10) )  )) ) - r)

@numba.jit(nopython=True)
def d_q(Ca,q):
    return (((1.0/(1.0+( 16/(Ca**4) )  )) ) - q)/450

@numba.jit(nopython=True)
def d_Ca(ICa,Ca):
    return (-0.13*ICa)-(Ca/13)

@numba.jit(nopython=True)
def d_c(V,c):
    return (((1.0/(1.0+( 1.0+np.exp(-(V+30.0)/(7) ) )  )) ) - c)/(2.0)

@numba.jit(nopython=True)
def d_h_den(V,h):
    return (((1.0/(1.0+( 1.0+np.exp((V+84.0)/(10.2) ) )  )) ) - h)/(0.1 + (1/(np.exp(-17.9-0.116*V) + np.exp(-1.84+0.09*V))))







#calcium current

@numba.jit(nopython=True)
def Ca1(V,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)-> float:

    t0 = gL*(V-VL)
    t1 = gNa * (((1.0/(1.0+np.exp(-(V+30.0)/9.5  )  )) )**3) * h * (V-VNa)
    t2 = gNaP *((1.0/(1.0+np.exp(-(V+47.0)/(3) )  )) ) * (V-VNa)
    t3 = gKdr * (n**4) * (V-VK)
    t4 = gA * (((1.0/(1.0+np.exp(-(V+50.0)/(20) )  )) )**3) * b*(V-VK)
    t5 = gM * z* (V-VK)
    t6 = ICa
    t7 = gC * ((1.0/(1.0+( 6/Ca )  )) ) * c * (V-VK)
    t8 = gsAHP * q * (V-VK)

    return (1/C)*(-t0-t1-t2-t3-t4-t5-t6-t7-t8+I+ISD)




#dendric equation

@numba.jit(nopython=True)
def DEN(V,VL,VH,h, g_L_den,g_h_den, I, IDS, C )-> float:
    
    t0 = g_L_den*(V-VL)
    t1 = g_h_den *h* (V-VH)

    return (1/C)*(-t0-t1+I+IDS)





#sub loop

@numba.jit(nopython=True)
def RK4_sub(func,V,y)-> float:
    K0 = 0.01*func(V,y)
    K1 = 0.01*func(V,y+K0/2.0)
    K2 = 0.01*func(V,y+K1/2.0)
    K3 = 0.01*func(V,y+K2)
    return y+(K0 + 2.0*K1 + 2.0*K2 + K3)/6





#dendric equation

@numba.jit(nopython=True)
def RK4_Ca1_den(func,V,VL,VH,h, g_L_den,g_h_den, I, IDS, C)-> float:
    K0 = 0.01*func(V,VL,VH,h, g_L_den,g_h_den, I, IDS, C)
    K1 = 0.01*func(V+K0/2.0,VL,VH,h, g_L_den,g_h_den, I, IDS, C)
    K2 = 0.01*func(V+K1/2.0,VL,VH,h, g_L_den,g_h_den, I, IDS, C)
    K3 = 0.01*func(V+K2,VL,VH,h, g_L_den,g_h_den, I, IDS, C)
    return V+ (K0 + 2.0*K1 + 2.0*K2 + K3)/6



#calcium current

@numba.jit(nopython=True)
def RK4_Ca1(func,V,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)-> float:
    K0 = 0.01*func(V,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)
    K1 = 0.01*func(V+K0/2.0,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)
    K2 = 0.01*func(V+K1/2.0,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)
    K3 = 0.01*func(V+K2,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)
    return V+ (K0 + 2.0*K1 + 2.0*K2 + K3)/6





#main loop


@numba.jit(nopython=True)
def RK4(V,V_s,V_d,T2,h,n,b,z,r,c,q,Ca, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK,gCa,gC,gsAHP,VCa,h_den, VH,gh,gL_den):
    
    Voltage = []
    Voltage.append(V_s)


    lit_h = []
    lit_n = []
    lit_b = []
    lit_z = []
    lit_r = []
    lit_c = []
    lit_h_den = []
    lit_ca = []
    lit_q = []
    lit_ica = []
    
    
    for i in range(len(T2)-1):

        I = T2[i]

        

        IDS = 0.2*(V_s-V_d)
        ISD = 0.2*(V_d-V_s)
        
        h=RK4_sub(d_h,V_s,h)
        n=RK4_sub(d_n,V_s,n)
        b=RK4_sub(d_b,V_s,b)
        z=RK4_sub(d_z,V_s,z)
        r=RK4_sub(d_r,V_s,r)
        c=RK4_sub(d_c,V_s,c)
        h_den = RK4_sub(d_h_den,V_d,h_den)

        ICa = gCa*(r**2)*(V_s-VCa)
        Ca = RK4_sub(d_Ca,ICa,Ca)
        q = RK4_sub(d_q,Ca,q)





        lit_h.append(h)
        lit_n.append(n)
        lit_b.append(b)
        lit_z.append(z)
        lit_r.append(r)
        lit_c.append(c)
        lit_h_den.append(h_den)
        lit_ca.append(Ca)
        lit_q.append(q)
        lit_ica.append(ICa)











        V_d = RK4_Ca1_den(DEN,V_d,VL,VH,h_den, gL_den,gh, I, IDS, C)


        V_s = RK4_Ca1(Ca1,V_s,I, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK , h, n, b, z,c, q, ICa, gC,gsAHP,Ca,ISD)

        Voltage.append(V_s)
    
    return [Voltage, lit_h ,  lit_n ,  lit_b,   lit_z , lit_r , lit_c , lit_h_den , lit_ca,  lit_q,  lit_ica ]





def two_comp_ca1(peram, T1,T2):

    #Parameter
    C = 1.0
    # gL = 0.05
    # VL = -70.0
    #gNa = 35.0
    #gNaP = 0.1 #0~0.41
    # gKdr = 6.0
    # gA = 1.4
    # gM = 1.0
    # VNa = 55.0
    # VK = -90
    
    # VH = 32.9
    # gh = 4.4
    # gL_den = 0.43

    # gCa = 0.2
    # gC = 10
    # gsAHP = 5
    # VCa = 120
    
    

    gL = peram[0]
    VL = peram[1]
    gNa = peram[2]
    gNaP = peram[3]
    gKdr = peram[4]
    gA = peram[5]
    gM = peram[6]
    VNa = peram[7]
    VK = peram[8]
    
    VH = peram[9]
    gh = peram[10]
    gL_den = peram[11]

    gCa = peram[12]
    gC = peram[13]
    gsAHP = peram[14]
    VCa = 120

    # gCa = 0.2
    # gC = 10
    # gsAHP = 5
    # VCa = 120
    

    #Initial Condition
    V = -72.0
    V_s = V
    V_d = V
    h,n,b,z,r,c,q,Ca = 0,0,0,0,0,0,0,0
    h_den=0

    #Applied Current
    #T1,T22 = stepcurr()
    #T1,T22 = squarewave1()
    #T1,T22 = squarewave2()
    #T1,T22 = sinewave()
    

    #T2 = np.array(T2)

    Voltage= RK4(V,V_s,V_d,T2,h,n,b,z,r,c,q,Ca, C ,gL ,VL ,gNa, gNaP,gKdr , gA , gM , VNa , VK,gCa,gC,gsAHP,VCa,h_den, VH,gh,gL_den)

    return Voltage