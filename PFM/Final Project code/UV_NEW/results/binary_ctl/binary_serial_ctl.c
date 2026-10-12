// Binary alloy (Ni-Cu) isothermal solidification WITH acoustic coupling (UV study)
// Based on: binary_serial.c (WBM-type entropy model, explicit FD + forward Euler)
//
// ACOUSTIC MODEL (Level 1 - oscillating undercooling, no flow coupling):
//   Travelling plane wave along x:  p_ac(x,t) = P_A * sin(omega*t - k*x)
//   Standing wave (MODE=2):          p_ac(x,t) = P_A * sin(k*x) * sin(omega*t)
//
// COUPLING (Clausius-Clapeyron melting-point shift, both components):
//   TmA_eff = Tm_A + dTmdp*P_A*sin(...)   [dimensionless: swing/T_ref]
//   TmB_eff = Tm_B + dTmdp*P_A*sin(...)
//   H_A/H_B/dH_A/dH_B use the effective melting points. T field stays
//   frozen (isothermal model). Concentration equation untouched directly;
//   solute responds only via the modified phi driving force.
//
// SCALE NOTES:
//   Variables here are DIMENSIONLESS (T_ref = 1728 K, l_ref = 1e-7 m).
//   Physical time per step = (4.6e-6)^2*0.1 = 2.116e-12 s; total ~2.1e-7 s.
//   Default f = 20 MHz -> ~4 cycles per run (scaled-frequency study).
//   P_A is a SENSITIVITY parameter (real 1 MPa -> ~0.025 K shift).
//
// HYGIENE: explicit phi=0 init, srand(UV_SEED). Snapshots ON (phi + c).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Binary alloy Ni and Cu
int main() {

    // Physical data (Ni) — identical to original binary_serial.c

    double sigma_A = 3.7e-5;  // surface energy (j/m^2)
    double beta_A = .33;      // kinetic coefficient (m/s/K)
    double Tm_A = 1728;     // Melting Temp
    double L_A = 2350;     // Latent heat (J/m^3)
    //double Cp_A = 5.42e6;     // Heat capacity(J/m^3/k)
    double D_lA=1e-5;         // Diffusitivity in liq (m^2/s)
    double D_sA=1e-9;       // Diffusitivity in solid (m^2/s)


    // Physical data (Cu)

    double sigma_B = 2.9e-5;
    double beta_B = .39;
    double Tm_B = 1358;
    double L_B = 1728;
    //double Cp_B = 3.96e6;
    double D_lB=1e-5;
    double D_sB=1e-9;


    int aniso = 4;
    double theta0 = 0;

    double d = 4.89e-6;    // interface width

    double M_A=Tm_A*beta_A/6/sqrt(2)/L_A/d;
    double M_B=Tm_B*beta_B/6/sqrt(2)/L_B/d;
    double w_A=3*sigma_A/sqrt(2)/Tm_A/d;
    double w_B=3*sigma_B/sqrt(2)/Tm_B/d;

   double eps0_A=6*sqrt(2)*sigma_A*d/Tm_A;



    double delta = 0.04;
    double alfa = 0.4;
    double vm=7.42;
    double R=8.314;
    // Define grid parameters
    int Nx = 100;
    int Ny = 100;

    double dx = 4.6e-6;
    double dy = dx;

    // Define time integration parameters
    double nstep = 1e5;
    double dt = dx*dx*.1;


    double T=1554.0;

    double T_ref=1728.0; // Referance temperature
    double l_ref=1e-7;   // referace lemgth

    // ============ ACOUSTIC / VIBRATION PARAMETERS (UV addition) ============
    int    UV_MODE   = 0;        // 0 = off (control), 1 = travelling, 2 = standing
    double UV_PA     = 4e8;      // pressure amplitude [Pa]; 4e8 -> ~10 K Tm swing
    double UV_FREQ   = 20e6;     // frequency [Hz]; 20 MHz -> ~4 cycles per run
    double UV_CSOUND = 4000.0;   // sound speed in liquid metal [m/s]
    double UV_DTMDP  = 2.5e-8;   // Clausius-Clapeyron slope dTm/dp [K/Pa]
    double UV_LAMBDA = -1.0;     // wavelength [m]; <0 means c_sound/f (physical)
    int    UV_SEED   = 7;        // RNG seed (reproducibility)
    // =======================================================================

    double uv_omega = 2 * M_PI * UV_FREQ;
    double uv_lambda = (UV_LAMBDA > 0) ? UV_LAMBDA : UV_CSOUND / UV_FREQ;
    double uv_k = 2 * M_PI / uv_lambda;
    double uv_dTm_nd = UV_DTMDP * UV_PA / T_ref;  // dimensionless Tm swing
    const double UV_DX_PHYS = 4.6e-6;             // physical grid spacing [m]
    const double UV_DT_PHYS = UV_DX_PHYS*UV_DX_PHYS*0.1;  // physical s per step

    printf("UV settings: mode=%d PA=%e Pa f=%e Hz lambda=%e m dTm_swing=%f K (nd %e) seed=%d\n",
           UV_MODE, UV_PA, UV_FREQ, uv_lambda, UV_DTMDP*UV_PA, uv_dTm_nd, UV_SEED);

    // Non Dimensionalization

    dx=dx/l_ref;
    dy=dx;
    dt=dt*D_lA/l_ref/l_ref;
    w_A=w_A/R/T_ref*vm;
    w_B=w_B/R/T_ref*vm;
    L_A=L_A/R/T_ref*vm;
    L_B=L_B/R/T_ref*vm;
    T=T/T_ref;
    Tm_A=Tm_A/T_ref;
    Tm_B=Tm_B/T_ref;

    M_A=M_A*l_ref*l_ref*R*T_ref/D_lA/vm;
    M_B=M_B*l_ref*l_ref*R*T_ref/D_lA/vm;

    eps0_A=eps0_A/l_ref*sqrt(vm/R/T_ref);

    D_lA=D_lA/D_lA;
    D_sA=D_sA/D_lA;

    D_lB=D_lB/D_lA;
    D_sB=D_sB/D_lA;



    // Initialize nucleus
    //double phi[Nx+10][Ny+10], T[Nx+10][Ny+10],c[Nx+10][Ny+10];
    double** phi = (double**)malloc(Nx * sizeof(double));
  //double** T = (double*)malloc(Nx * sizeof(double));
  double** c = (double**)malloc(Nx * sizeof(double));
   for (int i = 0; i < Nx; i++) {
        phi[i] = (double*)malloc(Ny * sizeof(double));
       // T[i] = (double*)malloc(Ny * sizeof(double));
        c[i] = (double*)malloc(Ny * sizeof(double));
    }

    srand(UV_SEED);   // UV: fixed seed (original left rand() unseeded)

    int r = 10; // radius of nucleus
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            double distance = sqrt((i) * (i) + (j) * (j));
            if (distance < r)
                {
                phi[i][j] = 1.0;

                }
            else
                {
                phi[i][j] = 0.0;   // UV: explicit zero (original: fresh-page zeros)
                }

           // T[i][j] = 1554.0;
            c[i][j]=.4083;
        }
    }

    // Evolve the system over time
    //double lap_phi[Nx + 10][Ny + 10],lap_c[Nx + 10][Ny + 10], lap_T[Nx + 10][Ny + 10], phi_dx[Nx + 10][Ny + 10], phi_dy[Nx + 10][Ny + 10], eps[Nx + 10][Ny + 10],d_eps[Nx + 10][Ny + 10];

    // Evolve the system over time
    double** lap_phi = (double**)malloc(Nx * sizeof(double));
    //double** lap_T = (double*)malloc(Nx * sizeof(double));
    double** lap_c = (double**)malloc(Nx * sizeof(double));
    double** phi_dx = (double**)malloc(Nx * sizeof(double));
    double** phi_dy = (double**)malloc(Nx * sizeof(double));
    double** c_dx = (double**)malloc(Nx * sizeof(double));
    double** c_dy = (double**)malloc(Nx * sizeof(double));
    double** eps = (double**)malloc(Nx * sizeof(double));
    double** d_eps = (double**)malloc(Nx * sizeof(double));
    for (int i = 0; i < Nx; i++) {
        lap_phi[i] = (double*)malloc(Ny * sizeof(double));
      //  lap_T[i] = (double*)malloc(Ny * sizeof(double));
        lap_c[i] = (double*)malloc(Ny * sizeof(double));
        phi_dx[i] = (double*)malloc(Ny * sizeof(double));
        phi_dy[i] = (double*)malloc(Ny * sizeof(double));
        c_dx[i] = (double*)malloc(Ny * sizeof(double));
        c_dy[i] = (double*)malloc(Ny * sizeof(double));
        eps[i] = (double*)malloc(Ny * sizeof(double));
        d_eps[i] = (double*)malloc(Ny * sizeof(double));
    }


    int n = 0;  //  Step Counter

    for (int i = 0; i < Nx; i++)
        {
            for (int j = 0; j < Ny; j++)
            {
                lap_phi[i][j] = 0.0;
                //lap_T[i][j] = 0.0;
                lap_c[i][j] = 0.0;
                phi_dx[i][j] = 0.0;
                phi_dy[i][j] = 0.0;
                eps[i][j] = 0.0;
                d_eps[i][j] = 0.0;
            }
        }
printf("Started");
    for (int iter = 1; iter <= nstep; iter++)
        {
            double t_now = iter * UV_DT_PHYS;   // UV: physical time for acoustic phase
            #pragma omp parallel for
            for(int i = 0; i < Nx; i++)
            {
                for(int j = 0; j < Ny; j++)
                {
                    int ip = i+1;
                    int im = i-1;
                    int jp = j+1;
                    int jm = j-1;

                    if(im == -1)
                    {
                        im = 0;
                    }
                    if(ip == Nx)
                    {
                        ip = Nx - 1;
                    }
                    if(jm == -1)
                    {
                        jm = 0;
                    }
                    if(jp == Ny)
                    {
                        jp = Ny - 1;
                    }

                    // Applying 5 point stencil
                    // Calculating Laplacian
                    double p1 = phi[ip][j]; // right
                    double p2 = phi[im][j]; // Left
                    double p3 = phi[i][jp]; // up
                    double p4 = phi[i][jm]; // down
                    double p5 = phi[i][j];  // centre

                    lap_phi[i][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                   // p1 = T[ip][j]; // right
                   // p2 = T[im][j]; // Left
                   // p3 = T[i][jp]; // up
                   // p4 = T[i][jm]; // down
                   // p5 = T[i][j];  // centre

                   // lap_T[i][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    p1 = c[ip][j]; // right
                    p2 = c[im][j]; // Left
                    p3 = c[i][jp]; // up
                    p4 = c[i][jm]; // down
                    p5 = c[i][j];  // centre

                    lap_c[i][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    // Calculating gradient

                    phi_dx[i][j] = (phi[ip][j] - phi[im][j]) / (2 * dx);
                    phi_dy[i][j] = (phi[i][jp] - phi[i][jm]) / (2 * dy);

                     c_dx[i][j] = (c[ip][j] - c[im][j]) / (2 * dx);
                    c_dy[i][j] = (c[i][jp] - c[i][jm]) / (2 * dy);


                    // Theta

                    double theta = atan2(phi_dy[i][j], phi_dx[i][j]);

                    // epsilon

                    eps[i][j] = eps0_A * (1 + delta * cos(aniso * (theta - theta0)));

                    // deps/dtheta

                    d_eps[i][j] = -eps0_A * delta * aniso * sin(aniso * (theta - theta0));
                }
            }
            // Time Evolution
            //#pragma omp parallel for
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    int ip = i + 1;
                    int im = i - 1;
                    int jp = j + 1;
                    int jm = j - 1;

                    if (im == -1) {
                        im = 0;
                    }
                    if (ip == Nx) {
                        ip = Nx - 1;
                    }
                    if (jm == -1) {
                        jm = 0;
                    }
                    if (jp == Ny) {
                        jp = Ny - 1;
                    }

                    // Time integration
                    double phi_old = phi[i][j];
                   //double T_old = T[i][j];
                   double c_old = c[i][j];


                    double p=phi_old*phi_old*phi_old*(6*phi_old*phi_old+15*phi_old-10);

                    double D=D_sA+p*(D_lA-D_sA);
                    //double M=(1-c[i][j])*M_A+c[i][j]*M_B;
                    double M=10;
                   // double K=(1-c[i][j])*K_A+c[i][j]*K_B;
                   // double Cp=(1-c[i][j])*Cp_A+c[i][j]*Cp_B;
                    //double L= (1-c[i][j])*L_A+c[i][j]*L_B;


                    double t1 = (eps[i][jp]*d_eps[i][jp]*phi_dx[i][jp] - eps[i][jm]*d_eps[i][jm]*phi_dx[i][jm]) / 2 / dy;
                    double t2 = (eps[ip][j]*d_eps[ip][j]*phi_dy[ip][j] - eps[im][j]*d_eps[im][j]*phi_dy[im][j]) / 2 / dx;


                    double g = phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                    double g_dash = 2 * phi_old * (1 - phi_old) * (1 - phi_old) - 2 * phi_old * phi_old * (1 - phi_old);
                    double g_ddash=2*(1 - phi_old) * (1 - phi_old)-8*phi_old * (1 - phi_old)+2*phi_old*phi_old;


                    // ===== UV ADDITION: acoustic pressure -> effective melting points =====
                    double TmA_eff = Tm_A, TmB_eff = Tm_B;
                    if (UV_MODE == 1) {          // travelling wave
                        double x = i * UV_DX_PHYS;
                        double dp = uv_dTm_nd * sin(uv_omega * t_now - uv_k * x);
                        TmA_eff = Tm_A + dp;
                        TmB_eff = Tm_B + dp;
                    } else if (UV_MODE == 2) {   // standing wave
                        double x = i * UV_DX_PHYS;
                        double dp = uv_dTm_nd * sin(uv_k * x) * sin(uv_omega * t_now);
                        TmA_eff = Tm_A + dp;
                        TmB_eff = Tm_B + dp;
                    }
                    double H_A=w_A*g_dash+30*g*L_A*(TmA_eff-T)/TmA_eff/T;
                    double H_B=w_B*g_dash+30*g*L_B*(TmB_eff-T)/TmB_eff/T;

                    double dH_A=w_A*g_ddash+30*g_dash*L_A*(TmA_eff-T)/TmA_eff/T;
                    double dH_B=w_B*g_ddash+30*g_dash*L_B*(TmB_eff-T)/TmB_eff/T;
                    // =====================================================================


                    double t3 =(1-c[i][j])*H_A+c[i][j]*H_B;

                    double dc_dp = 30*g*(D_lA-D_sA);


                    double random = -1 + (double)rand() / RAND_MAX * 2;
                    //double noise = 16 * alfa * random * phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                   double noise = -M*alfa*random*16*g*t3;



                    phi[i][j] = phi_old + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    //T[i][j] = T_old + dt * K/Cp * lap_T[i][j] + L / Cp * 30 * g * (phi[i][j] - phi_old);
                    //c[i][j]=c_old+dt*D*(lap_c[i][j]+vm/R*c[i][j]*(1-c[i][j])*(H_B-H_A)*lap_phi[i][j]);
                   /* double t4=(dc_dp+vm/R*D*(H_B-H_A)*(1-2*c_old))*(phi_dx[i][j]*c_dx[i][j]+phi_dy[i][j]*c_dy[i][j]);
                    double t5=D*c_old*(1-c_old)*vm/R*(H_B-H_A);
                    double t6=dc_dp*(H_B-H_A)+(dH_B-dH_A)*D;
                    double t7=vm/R*c_old*(1-c_old)*t6;*/

                    double t4=(dc_dp+D*(H_B-H_A)*(1-2*c_old))*(phi_dx[i][j]*c_dx[i][j]+phi_dy[i][j]*c_dy[i][j]);
                    double t5=D*c_old*(1-c_old)*(H_B-H_A);
                    double t6=dc_dp*(H_B-H_A)+(dH_B-dH_A)*D;
                    double t7=c_old*(1-c_old)*t6;


                    c[i][j]=c_old+dt*(D*lap_c[i][j]+t5*lap_phi[i][j]+t7*lap_phi[i][j]+t4);




                }

            }
         // printf(".%d", iter);

////////////////////////////////////// UV: snapshots ON (phi + c) /////////
if (((int)nstep >= 20000) && (iter % 20000 == 0)) {
            char fn1[48], fn2[48];
            sprintf(fn1, "bin_uv_phi_%d.csv", iter/20000);
            sprintf(fn2, "bin_uv_c_%d.csv", iter/20000);
            FILE *f1 = fopen(fn1, "w"), *f2 = fopen(fn2, "w");
            for (int a = 0; a < Nx; a++) {
                for (int b = 0; b < Ny; b++) {
                    fprintf(f1, "%e,", phi[a][b]);
                    fprintf(f2, "%e,", c[a][b]);
                }
                fprintf(f1, "\n");
                fprintf(f2, "\n");
            }
            fclose(f1);
            fclose(f2);
            printf("wrote %s %s at t=%e s\n", fn1, fn2, t_now);
          }
///////////////////////////////////////////////////////////////////////////

    }

for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            printf( "%e,    ", phi[i][j]);
        }

    }



char outphi[64], outc[64];
sprintf(outphi, "phib_uv_PA%.0e_f%.0e.csv", UV_PA, UV_FREQ);
sprintf(outc, "cb_uv_PA%.0e_f%.0e.csv", UV_PA, UV_FREQ);

    // open the file for writing
    FILE *fp = fopen(outphi, "w");

    // write to the excel file
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            fprintf(fp, "%e,    ", phi[i][j]);
        }
        fprintf(fp, "\n");
    }

    // close the file
    fclose(fp);




    // open the file for writing
    FILE *fp2 = fopen(outc, "w");

    // write to the excel file
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            fprintf(fp2, "%e,    ", c[i][j]);
        }
        fprintf(fp2, "\n");
    }

    // close the file
    fclose(fp2);


for(int i = 0; i < Nx; i++)
{
    free(phi[i]);
    //free(T[i]);
    free(c[i]);
    free(lap_phi[i]);
    //free(lap_T[i]);
    free(lap_c[i]);
    free(phi_dx[i]);
    free(phi_dy[i]);
    free(eps[i]);
    free(d_eps[i]);
    }

free(phi);
//free(T);
 free(c);
free(lap_phi);
//free(lap_T);
free(lap_c);
free(phi_dx);
free(phi_dy);
free(eps);
free(d_eps);


}
