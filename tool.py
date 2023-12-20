#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "AsifulHasan"
__copyright__ = "Copyright 2023, "
__date__ = "2023/11/23"
__version__ = "0.0.1"

#assumption - plane stress 
#           - balanced symmentric laminate

from math import e
import idna
import streamlit as st
import numpy as np

def tool():
    st.title("A Stress Analysis Tool for Laminated Composites")
    st.info(f"Change the parameters on the sidebar 👈 and press calculate", icon="ℹ️")

    st.sidebar.title("User Input Parameters")
    st.sidebar.divider()
    st.sidebar.title("Young Modulus")
    E1 = st.sidebar.number_input("E1 value", value=1.0)
    E2 = st.sidebar.number_input("E2 value", value=1.0)
    E3 = st.sidebar.number_input("E3 value", value=1.0)
    st.sidebar.title("Poisson Ratio")
    v12 = st.sidebar.number_input("ν12 value", value=0.2)
    v21 = v12
    v13 = st.sidebar.number_input("ν13 value", value=0.2)
    v23 = st.sidebar.number_input("ν23 value", value=0.2)
    st.sidebar.title("Shear Modulus")
    G12 = st.sidebar.number_input("G12 value", value=1.0)
    G13 = st.sidebar.number_input("G13 value", value=1.0)
    G23 = st.sidebar.number_input("G23 value", value=1.0)

    st.sidebar.header("Layup Sequence and Angles")
    layers = st.sidebar.number_input("Layer Numbers", value=3, min_value=1)
    theta = []
    for i in range(layers): # type: ignore
        theta.append(st.sidebar.number_input(f"Angle for Layer {i + 1}", value=0.0))

    st.sidebar.header("Thickness")
    h = st.sidebar.number_input(f"Thickness of 1 layer: ", value= 0.01)
    
    st.sidebar.header("Force resultant")
    Nx = st.sidebar.number_input("Enter force resultant in x direction- Nx", value=0.0)
    Ny = st.sidebar.number_input("Enter force resultant in y direction- Ny", value=0.0)
    Nxy = st.sidebar.number_input("Enter shear force resultant in xy- Nxy", value=0.0)

    st.sidebar.header("Moment resultant")
    Mx = st.sidebar.number_input("Enter moment resultant in x direction- Mx", value=0.0)
    My = st.sidebar.number_input("Enter moment resultant in y direction- My", value=0.0)
    Mxy = st.sidebar.number_input("Enter shear moment resultant in xy- Mxy", value=0.0)

    #st.sidebar.header("Coefficients of Thermal Expansion")
    #alpha1 = st.sidebar.number_input("Enter value of α1", value=0.0) 
    #alpha2 = st.sidebar.number_input("Enter value of α2", value=0.0)
    #alpha3 = st.sidebar.number_input("Enter value of α3", value=0.0)

    st.sidebar.title("Temperature")
    deltaT = st.sidebar.number_input("Enter value of ΔT", value=0.0)
    
    st.sidebar.title("Maximum Stresses")
    st.sidebar.header("Maximum Stresses in Compression")
    sigma_1c = st.sidebar.number_input("Enter value of σ1C", value=0.1)
    sigma_2c = st.sidebar.number_input("Enter value of σ2C", value=0.1)
    st.sidebar.header("Maximum Stresses in Tension")
    sigma_1t = st.sidebar.number_input("Enter value of σ1T", value=0.1)
    sigma_2t = st.sidebar.number_input("Enter value of σ2T", value=0.1)
    st.sidebar.header("Maximum Stresses in Shear")
    tao_12f = st.sidebar.number_input("Enter value of τ12F", value=0.1)
    
    
    

    if st.button("CALCUATE"):
        # Compliance Matrix
        S11 = 1 / E1
        S12 = -v12 / E1
        S13 = -v13 / E1
        S21 = -v12 / E1
        S22 = 1 / E2
        S23 = -v23 / E2
        S31 = -v13 / E1
        S32 = -v23 / E2
        S33 = 1 / E3
        S44 = 1 / G23
        S55 = 1 / G13
        S66 = 1 / G12

        # Compliance Matrix in matrix form
        st.subheader("The Compliance Matrix")
        SM = np.array([[S11, S12, S13, 0, 0, 0],
                       [S21, S22, S23, 0, 0, 0],
                       [S31, S32, S33, 0, 0, 0],
                       [0, 0, 0, S44, 0, 0],
                       [0, 0, 0, 0, S55, 0],
                       [0, 0, 0, 0, 0, S66]])

        st.write(SM)

        # Stiffness Matrix
        S = (S11 * S22 * S33) - (S11 * S23 * S23) - (S22 * S13 * S13) - (S33 * S12 * S12) + 2 * (S12 * S23 * S13)
        C11 = ((S22 * S33) - (S23 * S23)) / S
        C12 = ((S13 * S23) - (S12 * S33)) / S
        C22 = ((S33 * S11) - (S13 * S13)) / S
        C13 = ((S12 * S23) - (S13 * S22)) / S
        C33 = ((S11 * S22) - (S12 * S12)) / S
        C23 = ((S12 * S13) - (S23 * S11)) / S
        C44 = 1 / S44
        C55 = 1 / S55
        C66 = 1 / S66

        # Stiffness Matrix in matrix form
        st.subheader("The Stiffness matrix")
        CM = np.array([[C11, C12, C13, 0, 0, 0],
                       [C12, C22, C23, 0, 0, 0],
                       [C13, C23, C33, 0, 0, 0],
                       [0, 0, 0, C44, 0, 0],
                       [0, 0, 0, 0, C55, 0],
                       [0, 0, 0, 0, 0, C66]])

        st.write(CM)

        # Reduced Compliance Matrix
        st.subheader("The Reduced Compliance Matrix")
        Reduced_SM = np.array([[S11, S12, 0],
                               [S12, S22, 0],
                               [0, 0, S66]])

        st.write(Reduced_SM)

        # Reduced Stiffness Matrix
        Q11 = C11 - (C13 ** 2) / C33
        Q12 = C12 - (C13 * C23) / C33
        Q22 = C22 - (C23 ** 2) / C33
        Q66 = C66

        st.subheader("The Reduced Stiffness Matrix")
        Reduced_QM = np.array([[Q11, Q12, 0],
                               [Q12, Q22, 0],
                               [0, 0, Q66]])

        st.write(Reduced_QM)

        # angles = [+30, -30, 0, 0, -30, +30]
        # st.write(angles)
        #
        # st.write("m =")
        # for current_angle in angles:
        #     m_A = np.cos(np.radians(current_angle))
        #     st.write(m_A)
        #
        # st.write("n =")
        # for current_angle in angles:
        #     n_A = np.sin(np.radians(current_angle))
        # st.write(n_A)

        #m_A = np.cos(np.radians(theta))
        #n_A = np.sin(np.radians(theta))

        # User input for layer orientation angles
        # Calculate cosine and sine angles for each layer
        m = np.cos(np.degrees(theta))
        n = np.sin(np.degrees(theta))

        ## Transformed Reduced Compliance Matrix
        Sbar11 = np.zeros(layers)#type:ignore
        Sbar12 = np.zeros(layers)#type:ignore
        Sbar16 = np.zeros(layers)#type:ignore
        Sbar22 = np.zeros(layers)#type:ignore
        Sbar26 = np.zeros(layers)#type:ignore
        Sbar66 = np.zeros(layers)#type:ignore
        
        for i in range(layers): #type:ignore
            Sbar11[i] = S11 * m[i] ** 4 + ((2 * S12) + S66) * n[i] ** 2 * m[i] ** 2 + S22 * n[i] ** 4
            Sbar12[i] = (S11 + S22 - S66) * n[i] ** 2 * m[i] ** 2 + S12 * n[i] ** 4 + m[i] ** 4
            Sbar16[i] = (2 * S11 - 2 * S12 - S66) * n[i] * m[i] ** 3 - (2 * S22 - 2 * S12 - S66) * n[i] ** 3 * m[i]
            Sbar22[i] = S11 * n[i] ** 4 + (2 * S12 + S66) * n[i] ** 2 * m[i] ** 2 + S22 * m[i] ** 4
            Sbar26[i] = (2 * S11 - 2 * S12 - S66) * n[i] ** 3 * m[i] - (2 * S22 - 2 * S12 - S66) * n[i] * m[i] ** 3
            Sbar66[i] = 2 * (2 * S11 + 2 * S22 - 4 * S12 - S66) * n[i] ** 2 * m[i] ** 2 + S66 * (n[i] ** 4 + m[i] ** 4)

        # The transformed reduced compliance matrix in matrix form
        st.subheader("The Transformed Reduced Compliance Matrix")
        S_bar = np.array([[*Sbar11, *Sbar12, *Sbar16],
                          [*Sbar12, *Sbar22, *Sbar26],
                          [*Sbar16, *Sbar26, *Sbar66]])
        st.write(S_bar)
    

        # Transformed Reduced Stiffness Matrix
        Qbar11 = np.zeros(layers)# type: ignore
        Qbar12 = np.zeros(layers)# type: ignore
        Qbar16 = np.zeros(layers)# type: ignore
        Qbar22 = np.zeros(layers)# type: ignore
        Qbar26 = np.zeros(layers)# type: ignore
        Qbar66 = np.zeros(layers)# type: ignore

        for i in range(layers):# type: ignore
            Qbar11[i] = Q11 * m[i] ** 4 + (2 * Q12 + 2 * Q66) * n[i] ** 2 * m[i] ** 2 + Q22 * n[i] ** 4
            Qbar12[i] = (Q11 + Q22 - 4 * Q66) * n[i] ** 2 * m[i] ** 2 + Q12 * n[i] ** 4 + m[i] ** 4
            Qbar16[i] = (Q11 - Q12 - 2 * Q66) * n[i] * m[i] ** 3 + (Q12 - Q22 + 2 * Q66) * n[i] ** 3 * m[i]
            Qbar22[i] = Q11 * n[i] ** 4 + (2 * Q12 + 2 * Q66) * n[i] ** 2 * m[i] ** 2 + Q22 * m[i] ** 4
            Qbar26[i] = (Q11 - Q12 - 2 * Q66) * n[i] ** 3 * m[i] + (Q12 - Q22 + 2 * Q66) * n[i] * m[i] ** 3
            Qbar66[i] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * n[i] ** 2 * m[i] ** 2 + Q66 * (n[i] ** 4 + m[i] ** 4)

        # The transformed reduced stiffness matrix in matrix form
        st.subheader("The Transformed Reduced Stiffness Matrix")
        Q_bar = np.array([[*Qbar11, *Qbar12, *Qbar16],
                          [*Qbar12, *Qbar22, *Qbar26],
                          [*Qbar16, *Qbar26, *Qbar66]])

        st.write(Q_bar)
        
        # Thickness of the laminate
        Z = layers * h
        st.subheader(f'Thickness of the LAMINATE: {Z}')

        # ABD Matrix
        # QS = np.sum(Q_bar, axis=2)
        # QS = np.array([[sum(Qbar11)], Qbar12, Qbar16, Qbar22, Qbar26, Qbar66]).sum(axis=1)
        QS = [sum(Qbar11),
        sum(Qbar12),
        sum(Qbar16),
        sum(Qbar22),
        sum(Qbar26),
        sum(Qbar66)]
        

        # Matrix A
        A = np.zeros(6)
        for i in range(6):
            A[i] = np.sum(QS[i]) * h

        # Matrix B
        B = np.zeros(6)
        for i in range(6):
            B[i] = (1 / 2) * np.sum(QS[i]) *0* (h * (i + 1) ** 2 - h * (i) ** 2)

        # Matrix D
        D = np.zeros(6)
        for i in range(6):
            D[i] = (1 / 3) * np.sum(QS[i]) * (h * (i + 1) ** 3 - h * (i) ** 3)
            
        # ABD & abd Matrix
        st.subheader('The ABD matrix')
        ABD_m = np.array([[A[0], A[1], A[2], B[0], B[1], B[2]],
                          [A[1], A[3], A[4], B[1], B[3], B[4]],
                          [A[2], A[4], A[5], B[2], B[4], B[5]],
                          [B[0], B[1], B[2], D[0], D[1], D[2]],
                          [B[1], B[3], B[4], D[1], D[3], D[4]],
                          [B[2], B[4], B[5], D[2], D[4], D[5]]])
        st.write(ABD_m)
        
        # abd Matrix
        st.subheader('The abd matrix')
        abd_m = np.linalg.inv(ABD_m)
        st.write(abd_m)
        

# Engineering Properties of the laminate
        
        ExL = np.zeros(layers)# type: ignore
        VxyL = np.zeros(layers)# type: ignore
        EyL = np.zeros(layers)# type: ignore
        VyxL = np.zeros(layers)# type: ignore
        GxyL = np.zeros(layers)# type: ignore

        for i in range(layers):# type: ignore
            ExL[i] = E1 / (m[i] ** 4 + (E1 / G12 - 2 * v12) * n[i] ** 2 * m[i] ** 2 + (E1 / E2) * n[i] ** 4)
            
            VxyL[i] = (v12 * (n[i] ** 4 + m[i] ** 4) - (1 + E1 / E2 - (E1 / G12) * n[i] ** 2 * m[i] ** 2)) / (
                    m[i] ** 4 + (E1 / G12 - 2 * v12) * n[i] ** 2 * m[i] ** 2 + (E1 / E2) * n[i] ** 4)
            
            EyL[i] = E2 / (m[i] ** 4 + ((E2 / G12) - 2 * v12) * n[i] ** 2 * m[i] ** 2 + (E2 / E1) * n[i] ** 4)
            
            VyxL[i] = ((v21 * (n[i] ** 4 + m[i] ** 4)) - (1 + E2 / E1 - (E2 / G12) * n[i] ** 2 * m[i] ** 2)) / (
                    m[i] ** 4 + ((E2 / G12) - 2 * v21) * n[i] ** 2 * m[i] ** 2 + (E2 / E1) * n[i] ** 4)
            
            GxyL[i] = G12 / (n[i] ** 4 + m[i] ** 4 + 2 * (2 * (G12 / E1) * (1 + 2 * v12) + 2 * (G12 / E2) - 1) * n[i] ** 2 *
                        m[i] ** 2)

        st.subheader("Engineering Properties for Each Layer")
        for i in range(layers):# type: ignore
            st.write(f"**Layer {i + 1}**: Ex: {ExL[i]}, Vxy: {VxyL[i]}, Ey: {EyL[i]},Vyx: {VyxL[i]}, Gxy: {GxyL[i]}")


        # Effective engineering properties for a LAMINATE
        Ex_bar = (A[0] * A[3] - (A[1] ** 2)) / (A[3] * Z)
        Ey_bar = (A[0] * A[3] - (A[1] ** 2)) / (A[0] * Z)
        Gxy_bar = A[5] / Z
        vxy_bar = A[1] / A[3]
        vyx_bar = A[1] / A[0]

        # Display effective engineering properties
        st.subheader('Effective engineering properties for the laminate')
        st.write(f'Ex_bar: {Ex_bar}')
        st.write(f'Ey_bar: {Ey_bar}')
        st.write(f'Gxy_bar: {Gxy_bar}')
        st.write(f'vxy_bar: {vxy_bar}')
        st.write(f'vyx_bar: {vyx_bar}')

        # Calculation of coefficients a and d
        # Coefficients a
        det_A = (A[0] * (A[3] * A[5] - (A[4] ** 2))) - (A[1] * (A[1] * A[5] - A[4] * A[2])) + (
                    A[2] * (A[1] * A[4] - A[3] * A[2]))
        a = [
            (A[3] * A[5] - (A[4] ** 2)) / det_A,
            ((A[4] * A[2]) - (A[1] * A[5])) / det_A,
            ((A[1] * A[4]) - (A[3] * A[2])) / det_A,
            (A[0] * A[5] - (A[2] ** 2)) / det_A,
            ((A[1] * A[2]) - (A[0] * A[4])) / det_A,
            (A[0] * A[3] - (A[1] ** 2)) / det_A
        ]
        
        
        # Coefficients d
        det_D = (D[0] * (D[3] * D[5] - (D[4] ** 2))) - (D[1] * (D[1] * D[5] - D[4] * D[2])) + (
                    D[2] * (D[1] * D[4] - D[3] * D[2]))
        d = [
            (D[3] * D[5] - (D[4] ** 2)) / det_D,
            ((D[4] * D[2]) - (D[1] * D[5])) / det_D,
            ((D[1] * D[4]) - (D[3] * D[2])) / det_D,
            (D[0] * D[5] - (D[2] ** 2)) / det_D,
            ((D[1] * D[2]) - (D[0] * D[4])) / det_D,
            (D[0] * D[3] - (D[1] ** 2)) / det_D
        ]


        alpha1 = -0.001800 * 10**(-6)
        alpha2 =  24.3 * 10**(-6)
        alpha3 =  24.3 * 10**(-6)
        # Unit Thermal stress resultant
        st.subheader('Unit Thermal Stresses')
        alphaX = np.zeros(layers) # type: ignore 
        alphaY = np.zeros(layers) # type: ignore
        alphaXY = np.zeros(layers) # type: ignore

        for i in range(layers):# type: ignore
            alphaX[i] = alpha1 * m[i] ** 2 + alpha2 * n[i] ** 2
            alphaY[i] = alpha1 * n[i] ** 2 + alpha2 * m[i] ** 2
            alphaXY[i] = 2 * (alpha1 - alpha2) * m[i] * n[i]
            alphaZ = alpha3

        st.subheader('Values of coefficient of Thermal  expansion')
        st.write(f'alphaX: {alphaX}')
        st.write(f'alphaY: {alphaY}')
        st.write(f'alphaXY: {alphaXY}')

        # User input for temperature
        # deltaT = st.number_input('Enter the value of ΔT:', value=0.0)

        # The temperature is independent of the thickness. So we will have
        # different thermal stress resultant for each layer.

        st.subheader('Unit Thermal Stresses are')
        NxT_hat = np.zeros(layers)# type: ignore
        NyT_hat = np.zeros(layers)# type: ignore
        NxyT_hat = np.zeros(layers)# type: ignore
        MxT_hat = np.zeros(layers)# type: ignore
        MyT_hat = np.zeros(layers)# type: ignore
        MxyT_hat = np.zeros(layers)# type: ignore

        for i in range(layers):# type: ignore
            NxT_hat = (np.sum(Qbar11[i] * alphaX[i] + Qbar12[i] * alphaY[i] + Qbar16[i] * alphaXY[i])) * (
                        Z * (i + 1) - Z * i) * deltaT
            NyT_hat = (np.sum(Qbar12[i] * alphaX[i] + Qbar22[i] * alphaY[i] + Qbar26[i] * alphaXY[i])) * (
                        Z * (i + 1) - Z * i) * deltaT
            NxyT_hat = (np.sum(Qbar16[i] * alphaX[i] + Qbar26[i] * alphaY[i] + Qbar66[i] * alphaXY[i])) * (
                        Z * (i + 1) - Z * i) * deltaT
            MxT_hat = (1 / 2) * (np.sum(Qbar11[i] * alphaX[i] + Qbar12[i] * alphaY[i] + Qbar16[i] * alphaXY[i])) * (
                    Z * (i + 1) ** 2 - Z * i ** 2) * deltaT
            MyT_hat = (1 / 2) * (np.sum(Qbar12[i] * alphaX[i] + Qbar22[i] * alphaY[i] + Qbar26[i] * alphaXY[i])) * (
                    Z * (i + 1) ** 2 - Z * i ** 2) * deltaT
            MxyT_hat = (1 / 2) * (np.sum(Qbar16[i] * alphaX[i] + Qbar26[i] * alphaY[i] + Qbar66[i] * alphaXY[i])) * (
                    Z * (i + 1) ** 2 - Z * i ** 2) * deltaT

        # Print the results
        for i in range(layers):# type: ignore
            st.write(f'**Layer {i + 1}:**')
            st.write(f'NxT_hat: {NxT_hat}')
            st.write(f'NyT_hat: {NyT_hat}')
            st.write(f'NxyT_hat: {NxyT_hat}')
            st.write(f'MxT_hat: {MxT_hat}')
            st.write(f'MyT_hat: {MyT_hat}')
            st.write(f'MxyT_hat: {MxyT_hat}')

        # Extensional Strain
        
        epsilon_x0 = a[0] * (Nx+NxT_hat) + a[1] * (Ny+NyT_hat) + a[2] * (Nxy+NxyT_hat)
        epsilon_y0 = a[1] * (Nx+NxT_hat) + a[3] * (Ny+NyT_hat) + a[4] * (Nxy+NxyT_hat)
        gama_xy0 = a[2] * (Nx+NxT_hat) + a[4] * (Ny+NyT_hat)+ + a[5] * (Nxy+NxyT_hat)
        st.subheader('Extensional strain')
        st.write(f'epsilon_x0: {epsilon_x0}')
        st.write(f'epsilon_y0: {epsilon_y0}')
        st.write(f'gama_xy0: {gama_xy0}')

        # Curvature
        # st.subheader('Curvature')
        Kx_o = d[0] * (Mx+MxT_hat) + d[1] * (My+MyT_hat) + d[2] * (Mxy+MxyT_hat)
        Ky_o = d[1] * (Mx+MxT_hat) + d[3] * (My+MyT_hat) + d[4] * (Mxy+MxyT_hat)
        Kxy_o = d[2] * (Mx+MxT_hat) + d[4] * (My+MyT_hat) + d[5] * (Mxy+MxyT_hat)
        st.subheader('Curvature:')
        st.write(f'Kx_o: {Kx_o}')
        st.write(f'Ky_o: {Ky_o}')
        st.write(f'Kxy_o: {Kxy_o}')

        # Stress and Strain calculation
        epsilon_x = epsilon_x0 + Z * Kx_o
        epsilon_y = epsilon_y0 + Z * Ky_o
        gama_xy = gama_xy0 + Z * Kxy_o

        # Display strains in global coordinate system
        st.subheader('Strains in global coordinate system:')
        st.write(f'epsilon_x: {epsilon_x}')
        st.write(f'epsilon_y: {epsilon_y}')
        st.write(f'gama_xy: {gama_xy}')
        
        # Form the stress-strain relationship to find the stress in the global coordinate system
        sigma_x = np.zeros(layers)# type: ignore
        sigma_y = np.zeros(layers)# type: ignore
        tao_xy = np.zeros(layers)# type: ignore
        for i in range(layers):#type:ignore
            sigma_x[i] = Qbar11[i] * epsilon_x + Qbar12[i] * epsilon_y + Qbar16[i] * gama_xy
            sigma_y[i] = Qbar12[i] * epsilon_x + Qbar22[i] * epsilon_y + Qbar26[i] * gama_xy
            tao_xy[i] = Qbar16[i] * epsilon_x + Qbar26[i] * epsilon_y + Qbar66[i] * gama_xy
        
        st.subheader('Stress in the global coordinate system')
        for i in range(layers):#type:ignore
            st.write(f'**layer {i + 1}:**')
            st.write(f'sigma_x: {sigma_x[i]}')
            st.write(f'sigma_y: {sigma_y[i]}')
            st.write(f'tao_xy: {tao_xy[i]}')
       
        # Principal Stress Calculation
        sigma_1 = np.zeros(layers)# type: ignore
        sigma_2 = np.zeros(layers)# type: ignore
        tao_12 = np.zeros(layers)# type: ignore
        
        for i in range(layers):# type: ignore
            sigma_1[i] = m[i] ** 2 * sigma_x[i] + n[i] ** 2 * sigma_y[i] + 2 * m[i] * n[i] * tao_xy[i]
            sigma_2[i] = n[i] ** 2 * sigma_x[i] + m[i] ** 2 * sigma_y[i] - 2 * m[i] * n[i] * tao_xy[i]
            tao_12[i] = -n[i] * m[i] * sigma_x[i] + n[i] * m[i] * sigma_y[i] + (m[i] ** 2 - n[i] ** 2) * tao_xy[i]

        st.subheader('Stress in the principal coordinate system')
        for i in range(layers):#type:ignore
            st.write(f'**layer{i + 1}:**')
            st.write(f'sigma_1: {sigma_1[i]}')
            st.write(f'sigma_2: {sigma_2[i]}')
            st.write(f'tao_12: {tao_12[i]}')
            

        # Construct the principal stress matrix
        #principle_stress = np.array([sigma_1, sigma_2, tao_12])

        # Print the result
        #st.subheader('Principal Stress Matrix')
        #st.write(principle_stress)
        
        #
        alpha1 = -0.001800 * 10**(-6)
        alpha2 =  24.3 * 10**(-6)
        alpha3 =  24.3 * 10**(-6)
        # Unit Thermal stress resultant
        st.subheader('Unit Thermal Stresses')
        alphaX = np.zeros(layers) # type: ignore 
        alphaY = np.zeros(layers) # type: ignore
        alphaXY = np.zeros(layers) # type: ignore

        for i in range(layers):# type: ignore
            alphaX[i] = alpha1 * m[i] ** 2 + alpha2 * n[i] ** 2
            alphaY[i] = alpha1 * n[i] ** 2 + alpha2 * m[i] ** 2
            alphaXY[i] = 2 * (alpha1 - alpha2) * m[i] * n[i]
            alphaZ = alpha3

        st.subheader('Values of coefficient of Thermal  expansion')
        st.write(f'alphaX: {alphaX}')
        st.write(f'alphaY: {alphaY}')
        st.write(f'alphaXY: {alphaXY}')

        # User input for temperature
        # deltaT = st.number_input('Enter the value of ΔT:', value=0.0)

        # The temperature is independent of the thickness. So we will have
        # different thermal stress resultant for each layer.


        
        # Print the results
    
        # Maximum Stress Criteria
        #sigma_1c = -1.250
        #sigma_1t = 1.50
        #sigma_2c = -0.200
        #sigma_2t = 0.050
        #tao_12f = 0.100
        

        st.subheader('Maximum Stress Failure mode')
        st.subheader('Tension')
        for i in range(layers):# type: ignore
            if abs(sigma_1[i]) > abs(sigma_1c) or abs(sigma_1[i]) < abs(sigma_1t):
                st.write(f'sigma1: {sigma_1[i]}')
                st.write(f'sigma1c: {sigma_1c}')
                st.write(f'sigma1t: {sigma_1t}')
                
                st.write(f'**Layer {i + 1}:** Fail in Tension')
            else:
                st.write(f'**Layer {i + 1}:** Safe in Tension')

        st.subheader('Compression')
        for i in range(layers):# type: ignore
            if abs(sigma_2[i]) > abs(sigma_2c) or abs(sigma_2[i]) < abs(sigma_2t):
                st.write(f'sigma2: {sigma_2[i]}')
                st.write(f'sigma2c: {sigma_2c}')
                st.write(f'sigma2t: {sigma_2t}')
                
                st.write(f'**Layer {i + 1}:** Fail in Compression')
            else:
                st.write(f'**Layer {i + 1}:** Safe in Compression')

        st.subheader('Shear')
        for i in range(layers):# type: ignore
            if abs(tao_12[i]) >= abs(-tao_12f) or abs(tao_12[i]) >= abs(tao_12f):
                st.write(f'tao12: {tao_12[i]}')
                st.write(f'-tao12f: {-tao_12f}')
                st.write(f'tao12f: {tao_12f}')
                st.write(f'**Layer {i + 1}:** Fail in Shear')
            else:
                st.write(f'**Layer {i + 1}:** Safe in Shear')
                
                
                
        # T-Sai Wu Failiure Criteria
        #Analysis of T-SaiWu Failure Criteria. 
        st.header('Tsai-Wu Failure Mode')
        F1 = ((1/sigma_1t) + (1/sigma_1c))
        F2 = ((1/sigma_2t) + (1/sigma_2c))
        F11 = (-1/sigma_1c * sigma_1t)
        F22 = (-1/sigma_2c * sigma_2t)
        F66 = (1/tao_12f) ** 2
    
        st.subheader('coeffiecient F')
        st.write(f'F1: {F1}')
        st.write(f'F2: {F2}')
        st.write(f'F11: {F11}')
        st.write(f'F22: {F22}')
        st.write(f'F66: {F66}')
    
        
        for i in range(layers):#type:ignore
    
          if abs((F1 * sigma_1[i]) + (F2 * sigma_2[i]) + (F11 * (sigma_1[i] ** 2)) + (F22 * (sigma_2[i] ** 2)) + (F22 * (tao_12[i] ** 2)) - ((np.sqrt(F11 * F22)) * sigma_1[i] * sigma_2[i])) >= 1:
           st.write(f'**Layer {i + 1}:** Fail according to Tsai Wu criteria')
          else:
           st.write(f'**Layer {i + 1}:** Safe according to Tsai Wu Criteria') 

    st.toast('Done, Hooray!', icon='🎉')
