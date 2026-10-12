// C Code for dendrite arm spacing WITH acoustic/vibration coupling (UV study)
// Based on: Arm_Spacing.c (Kim pure-Ni model, explicit finite difference + forward Euler)
// Generates final phi field for post-processing (plot in MATLAB, measure arms).
//
// ACOUSTIC MODEL (Level 1 - oscillating undercooling, no flow coupling):
//   Travelling plane wave along x:  p_ac(x,t) = P_A * sin(omega*t - k*x)
//   Standing wave (MODE=2):          p_ac(x,t) = P_A * sin(k*x) * sin(omega*t)
//   omega = 2*pi*f,  k = 2*pi/lambda_ac,  lambda_ac default = c_sound / f
//
// COUPLING (Clausius-Clapeyron melting-point shift):
//   Tm_eff(x,t) = Tm + dTmdp * p_ac(x,t)
//   Only the thermal driving term t3 uses Tm_eff:
//     t3 = w*g_dash + 30*g*(H/Tm)*(T_old - Tm_eff)
//
// PHYSICAL SCALE NOTE:
//   Real dTmdp for Ni is ~2.5e-8 K/Pa, so a real 1 MPa ultrasound field
//   shifts Tm by ~0.025 K (invisible vs 200 K undercooling). P_A here is a
//   SENSITIVITY parameter: P_A = 4e8 Pa -> ~10 K swing, 1.2e9 Pa -> ~30 K.
//   Real refining mechanisms (cavitation, streaming) are NOT in this file.
//
// TIME SCALE NOTE:
//   Total run = nstep*dt = 5e-7 s. Default f = 10 MHz gives ~5 cycles per run
//   (scaled-frequency sensitivity study; a real 20 kHz wave would look frozen).
//
// NUMERICAL HYGIENE (no physics change vs original):
//   - phi explicitly zero-initialised (original relied on zeroed fresh pages)
//   - srand(UV_SEED) fixed seed so runs are reproducible
//
// OUTPUT: uv_space_<n>.csv snapshots every 50000 steps + final
//   space_uv_PA<f>_f<f>.csv (filenames embed the acoustic settings).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main() {

    // Define model parameters (identical to original Arm_Spacing.c)
    double sigma0 = 0.37;
    double nu = 2;
    double Tm = 1728;
    double H = 2.35e9;
    double D = 1.55e-5;
    double Cp = 5.42e6;
    double lambda = 4e-8;

    int aniso = 4;
    double theta0 = 0;

    // Define other parameters
    double w = 6.6 * sigma0 / lambda;
    double eps0 = sqrt(2.73 * lambda * sigma0);
    double M = Tm * nu / 2.73 / H / lambda;
    double delta = 0.5;
    double alfa = 0.05;

    // Define grid parameters
    int Nx = 400;
    int Ny = 400;

    double dx = 2e-8;
    double dy = 2e-8;

    // Define time integration parameters
    int nstep = 5e5;
    double dt = 1e-12;

    // ============ ACOUSTIC / VIBRATION PARAMETERS (UV addition) ============
    int    UV_MODE   = 2;        // 0 = off (control), 1 = travelling, 2 = standing
    double UV_PA     = 1.2e9;      // pressure amplitude [Pa]; 4e8 -> ~10 K Tm swing
    double UV_FREQ   = 10e6;     // frequency [Hz]; 10 MHz -> ~5 cycles per run
    double UV_CSOUND = 4000.0;   // sound speed in liquid metal [m/s]
    double UV_DTMDP  = 2.5e-8;   // Clausius-Clapeyron slope dTm/dp [K/Pa], Ni estimate
    double UV_LAMBDA = -1.0;     // wavelength [m]; <0 means c_sound/f (physical)
    int    UV_SEED   = 7;        // RNG seed (reproducibility)
    // =======================================================================

    double uv_omega = 2 * M_PI * UV_FREQ;
    double uv_lambda = (UV_LAMBDA > 0) ? UV_LAMBDA : UV_CSOUND / UV_FREQ;
    double uv_k = 2 * M_PI / uv_lambda;
    double uv_dTm = UV_DTMDP * UV_PA;   // Tm swing amplitude [K] for the log

    printf("UV settings: mode=%d PA=%e Pa f=%e Hz lambda=%e m dTm_swing=%f K seed=%d\n",
           UV_MODE, UV_PA, UV_FREQ, uv_lambda, uv_dTm, UV_SEED);

  // double phi[Nx+10][Ny+10], T[Nx+10][Ny+10];
   double** phi = (double**)malloc(Nx * sizeof(double*));
   double** T = (double**)malloc(Nx * sizeof(double*));
   for (int i = 0; i < Nx; i++) {
        phi[i] = (double*)malloc(Ny * sizeof(double));
        T[i] = (double*)malloc(Ny * sizeof(double));
    }


/////////////////////////////////////////////     Nucleation     //////////////////////////////////////////

    srand(UV_SEED);   // UV: fixed seed (original left rand() unseeded)

    int r = 5; // radius of nucleus
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
           // double distance = sqrt((i-Nx) * (i-Nx) + (j-Ny/2) * (j-Ny/2));

            //if (distance < r)
               // {
               // phi[i][j] = 1.0;
               // }
                phi[Nx-i-1][j]=1.0;       // Thin solid layer at the bottom

        }
    }
    // UV: explicitly zero the rest (original relied on fresh-page zeros)
    for (int i = 0; i < Nx - 5; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            phi[i][j] = 0.0;
        }
    }

    for(int j=0;j<Nx;j++)
    {
    for (int i=0;i<Ny; i++)
    {
        T[j][i]=Tm-200;             // Supercooling
    }
    }

/////////////////////////////////////////////////////////////////////////////////////////




   //double lap_phi[Nx + 10][Ny + 10], lap_T[Nx + 10][Ny + 10], phi_dx[Nx + 10][Ny + 10], phi_dy[Nx + 10][Ny + 10], eps[Nx + 10][Ny + 10],d_eps[Nx + 10][Ny + 10];
    // Evolve the system over time
   double** lap_phi = (double**)malloc(Nx * sizeof(double*));
    double** lap_T = (double**)malloc(Nx * sizeof(double*));
   double** phi_dx = (double**)malloc(Nx * sizeof(double*));
   double** phi_dy = (double**)malloc(Nx * sizeof(double*));
   double** eps = (double**)malloc(Nx * sizeof(double*));
   double** d_eps = (double**)malloc(Nx * sizeof(double*));
    for (int i = 0; i < Nx; i++) {
        lap_phi[i] = (double*)malloc(Ny * sizeof(double));
        lap_T[i] = (double*)malloc(Ny * sizeof(double));
        phi_dx[i] = (double*)malloc(Ny * sizeof(double));
        phi_dy[i] = (double*)malloc(Ny * sizeof(double));
        eps[i] = (double*)malloc(Ny * sizeof(double));
        d_eps[i] = (double*)malloc(Ny * sizeof(double));
    }


    int n = 0;  //  Step Counter

    for (int i = 0; i < Nx; i++)
        {
            for (int j = 0; j < Ny; j++)
            {
                lap_phi[i][j] = 0.0;
                lap_T[i][j] = 0.0;
                phi_dx[i][j] = 0.0;
                phi_dy[i][j] = 0.0;
                eps[i][j] = 0.0;
                d_eps[i][j] = 0.0;
            }
        }

    for (int iter = 1; iter <= nstep; iter++)
        {
            double t_now = iter * dt;   // UV: physical time for acoustic phase

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

                    p1 = T[ip][j]; // right
                    p2 = T[im][j]; // Left
                    p3 = T[i][jp]; // up
                    p4 = T[i][jm]; // down
                    p5 = T[i][j];  // centre

                    lap_T[i][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    // Calculating gradient

                    phi_dx[i][j] = (phi[ip][j] - phi[im][j]) / (2 * dx);
                    phi_dy[i][j] = (phi[i][jp] - phi[i][jm]) / (2 * dy);

                    // Theta

                    double theta = atan2(phi_dy[i][j], phi_dx[i][j]);

                    // epsilon

                    eps[i][j] = eps0 * (1 + delta * cos(aniso * (theta - theta0)));

                    // deps/dtheta

                    d_eps[i][j] = -eps0 * delta * aniso * sin(aniso * (theta - theta0));
                }
            }
            // Time Evolution
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
                    double T_old=T[i][j];
                    double t1 = (eps[i][jp]*d_eps[i][jp]*phi_dx[i][jp] - eps[i][jm]*d_eps[i][jm]*phi_dx[i][jm]) / 2 / dy;
                    double t2 = (eps[ip][j]*d_eps[ip][j]*phi_dy[ip][j] - eps[im][j]*d_eps[im][j]*phi_dy[im][j]) / 2 / dx;
                    double g = phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                    double g_dash = 2 * phi_old * (1 - phi_old) * (1 - phi_old) - 2 * phi_old * phi_old * (1 - phi_old);
                    double random = -1 + (double)rand() / RAND_MAX * 2;
                    double noise = 16 * alfa * random * phi_old * phi_old * (1 - phi_old) * (1 - phi_old);

                    // ===== UV ADDITION: acoustic pressure -> local melting point =====
                    double Tm_eff = Tm;
                    if (UV_MODE == 1) {          // travelling wave
                        double x = i * dx;
                        Tm_eff = Tm + UV_DTMDP * UV_PA * sin(uv_omega * t_now - uv_k * x);
                    } else if (UV_MODE == 2) {   // standing wave
                        double x = i * dx;
                        Tm_eff = Tm + UV_DTMDP * UV_PA * sin(uv_k * x) * sin(uv_omega * t_now);
                    }
                    double t3 = w * g_dash + 30 * g * H / Tm * (T_old - Tm_eff);
                    // =================================================================
                    phi[i][j] = phi[i][j] + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    T[i][j] = T_old + dt * D * lap_T[i][j] + H / Cp * 30 * g * (phi[i][j] - phi_old);

                }

            }
//printf(".%d", iter);


////////////////////////////////////// UV: snapshots ON (were commented out) /////////

if (iter % 50000 == 0) {
            char filename[40];
            sprintf(filename, "uv_space_%d.csv", iter/50000);
            FILE* fp = fopen(filename, "w");
            for (int j = 0; j < Nx; j++) {
                for (int k = 0; k < Ny; k++) {
                    fprintf(fp, "%e,", phi[j][k]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("wrote %s at t=%e s\n", filename, t_now);
          }

///////////////////////////////////////////////////////////////////////////




    }

    char outname[64];
    sprintf(outname, "space_uv_PA%.0e_f%.0e.csv", UV_PA, UV_FREQ);

    // open the file for writing
    FILE *fp1 = fopen(outname, "w");

    // write to the excel file
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            fprintf(fp1, "%e,    ", phi[i][j]);
        }
        fprintf(fp1, "\n");
    }

    // close the file
    fclose(fp1);



for(int i = 0; i < Nx; i++)
{
    free(phi[i]);
    free(T[i]);

    free(lap_phi[i]);
    free(lap_T[i]);

    free(phi_dx[i]);
    free(phi_dy[i]);
    free(eps[i]);
    free(d_eps[i]);
    }

free(phi);
free(T);

free(lap_phi);
free(lap_T);

free(phi_dx);
free(phi_dy);
free(eps);
free(d_eps);

}
