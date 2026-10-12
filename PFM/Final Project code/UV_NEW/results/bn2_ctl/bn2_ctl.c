// Non dimensional binary Cu-Ni model


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// count cells with phi>0.5
static double count_phi(double** p, int nx, int ny){ double s=0; for(int a=0;a<nx;a++) for(int b=0;b<ny;b++) if(p[a][b]>0.5) s++; return s; }

// Binary alloy Ni and Cu
int main() {

    // Physical data (Ni)

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
    double theta0 = M_PI/4;

    double d = 4.89e-6;    // interface width
    double lambda_A=d/2;  // interface width/2
    double lambda_B=d/2;
    // Define model parameters

    /*double w_A = 6.6 * sigma_A / lambda_A;
    double eps0_A = sqrt(2.73 * lambda_A * sigma_A);
    double M_A = Tm_A * beta_A / 2.73 / L_A / lambda_A;

    double w_B = 6.6 * sigma_B/ lambda_B;
    double eps0_B = sqrt(2.73 * lambda_B * sigma_B);
    double M_B = Tm_B * beta_B / 2.73 / L_B / lambda_B;*/

//    printf("%e   %e   %e    \n",w_A,M_A,eps0_A);
   // printf("%e   %e   %e    \n",w_B,M_B,eps0_B);

    double M_A=Tm_A*beta_A/6/sqrt(2)/L_A/d;
    double M_B=Tm_B*beta_B/6/sqrt(2)/L_B/d;
    double w_A=3*sigma_A/sqrt(2)/d;
    double w_B=3*sigma_B/sqrt(2)/d;

   double eps0_A=sqrt(6*sqrt(2)*sigma_A*d);



    double delta = 0.04;
    double alfa = 0.4;
    double vm=7.42;
    double R=8.314;
    // Define grid parameters
    int Nx =100/2;
    int Ny =100/2;

    double dx = 4.6e-6;
    double dy = dx;

    // Define time integration parameters
    double nstep = 2e6;
    double dt = dx*dx*.1;


    double T=1574.0;

    double T_ref=1728.0; // Referance temperature
    double l_ref=1e-7;   // referace lemgth

    // ============ ACOUSTIC / VIBRATION PARAMETERS (UV addition) ============
    int    UV_MODE   = 0;        // 0 = off (control), 1 = travelling, 2 = standing
    double UV_PA     = 4e8;      // pressure amplitude [Pa]
    double UV_FREQ   = 1e6;      // frequency [Hz]; ~4 cycles per 2e6-step run
    double UV_CSOUND = 4000.0;   // sound speed in liquid metal [m/s]
    double UV_DTMDP  = 2.5e-8;   // Clausius-Clapeyron slope dTm/dp [K/Pa]
    double UV_LAMBDA = -1.0;     // wavelength [m]; <0 means c_sound/f
    int    UV_SEED   = 7;        // RNG seed
    int    CONS_MODE = 1;        // 0 = legacy coeff*lap (non-conservative), 1 = flux-divergence form
    // =======================================================================
    int    STREAM    = 0;        // Level-2 streaming: 0 off, 1 on
    double UV_VEL    = 0.0;      // streaming velocity amplitude [m/s]
    double uv_omega = 2 * M_PI * UV_FREQ;
    double uv_lambda = (UV_LAMBDA > 0) ? UV_LAMBDA : UV_CSOUND / UV_FREQ;
    double uv_k = 2 * M_PI / uv_lambda;
    double uv_dTm_nd = UV_DTMDP * UV_PA / T_ref;
    const double UV_DX_PHYS = 4.6e-6;
    const double UV_DT_PHYS = UV_DX_PHYS*UV_DX_PHYS*0.1;
    printf("UV settings: mode=%d PA=%e Pa f=%e Hz dTm_swing=%f K seed=%d\n",
           UV_MODE, UV_PA, UV_FREQ, UV_DTMDP*UV_PA, UV_SEED);
    printf("CONS_MODE=%d  STREAM=%d VEL=%e m/s\n", CONS_MODE, STREAM, UV_VEL);


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

 printf("%e   %e   %e    \n",w_A,M_A,eps0_A);
    printf("%e   %e   %e    \n",w_B,M_B,eps0_A);

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

srand(UV_SEED);   // UV: fixed seed

    int r = 10; // radius of nucleus
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            double distance = sqrt((i) * (i) + (j) * (j));

            if (distance < r)
                {
                phi[i][j] = 1.0;
                //c[i][j]=.86;
                }
                else
                {
                phi[i][j] = 0.0;   // UV: explicit zero
                }

           // T[i][j] = 1554.0;

            c[i][j]=.40381;
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
    double** Dco  = (double**)malloc(Nx * sizeof(double));
    double** Fc   = (double**)malloc(Nx * sizeof(double));
    double** dfc  = (double**)malloc(Nx * sizeof(double));
    double** dfp  = (double**)malloc(Nx * sizeof(double));
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
        Dco[i] = (double*)malloc(Ny * sizeof(double));
        Fc[i]  = (double*)malloc(Ny * sizeof(double));
        dfc[i] = (double*)malloc(Ny * sizeof(double));
        dfp[i] = (double*)malloc(Ny * sizeof(double));
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
                Dco[i][j] = 0.0;
                Fc[i][j] = 0.0;
                dfc[i][j] = 0.0;
                dfp[i][j] = 0.0;
            }
        }
printf("Started");
    for (int iter = 1; iter <= nstep; iter++)
        {
            double t_now = iter * UV_DT_PHYS;   // UV physical time
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

                    // ---- conservative-form coefficients (computed on OLD phi,c) ----
                    double phi_o = phi[i][j];
                    double c_o   = c[i][j];
                    double pp = phi_o*phi_o*phi_o*(6*phi_o*phi_o+15*phi_o-10);
                    double Dc_o = D_sA + pp*(D_lA - D_sA);
                    double gg = phi_o*phi_o*(1-phi_o)*(1-phi_o);
                    double gd = 2*phi_o*(1-phi_o)*(1-phi_o) - 2*phi_o*phi_o*(1-phi_o);
                    double gdd = 2*(1-phi_o)*(1-phi_o) - 8*phi_o*(1-phi_o) + 2*phi_o*phi_o;
                    double TmA_e = Tm_A, TmB_e = Tm_B;
                    if (UV_MODE == 1) {
                        double xx = i * UV_DX_PHYS;
                        double dpv = uv_dTm_nd * sin(uv_omega * t_now - uv_k * xx);
                        TmA_e = Tm_A + dpv; TmB_e = Tm_B + dpv;
                    } else if (UV_MODE == 2) {
                        double xx = i * UV_DX_PHYS;
                        double dpv = uv_dTm_nd * sin(uv_k * xx) * sin(uv_omega * t_now);
                        TmA_e = Tm_A + dpv; TmB_e = Tm_B + dpv;
                    }
                    double HA_e = w_A*gd + 30*gg*L_A*(TmA_e-T)/TmA_e/T;
                    double HB_e = w_B*gd + 30*gg*L_B*(TmB_e-T)/TmB_e/T;
                    double dHA_e = w_A*gdd + 30*gd*L_A*(TmA_e-T)/TmA_e/T;
                    double dHB_e = w_B*gdd + 30*gd*L_B*(TmB_e-T)/TmB_e/T;
                    double dcdp_e = 30*gg*(D_lA - D_sA);
                    double HBA = HB_e - HA_e;
                    double t5_e = Dc_o*c_o*(1-c_o)*vm/R*HBA;
                    double t6_e = dcdp_e*HBA + (dHB_e - dHA_e)*Dc_o;
                    double t7_e = vm/R*c_o*(1-c_o)*t6_e;
                    Dco[i][j] = Dc_o;
                    Fc[i][j]  = t5_e + t7_e;
                }
            }
            // ---- Loop A2: conservative flux divergences on OLD state ----
            if (CONS_MODE) {
                #pragma omp parallel for
                for (int i = 0; i < Nx; i++) {
                    for (int j = 0; j < Ny; j++) {
                        int ip = i+1, im = i-1, jp = j+1, jm = j-1;
                        if (im==-1) im=0;  if (ip==Nx) ip=Nx-1;
                        if (jm==-1) jm=0;  if (jp==Ny) jp=Ny-1;
                        // div(D grad c)
                        double Dfp = 0.5*(Dco[i][j]+Dco[ip][j]);
                        double Dfm = 0.5*(Dco[i][j]+Dco[im][j]);
                        double Dgp = 0.5*(Dco[i][j]+Dco[i][jp]);
                        double Dgm = 0.5*(Dco[i][j]+Dco[i][jm]);
                        dfc[i][j] = ( Dfp*(c[ip][j]-c[i][j]) - Dfm*(c[i][j]-c[im][j]) )/(dx*dx)
                                  + ( Dgp*(c[i][jp]-c[i][j]) - Dgm*(c[i][j]-c[i][jm]) )/(dy*dy);
                        // div(F grad phi)
                        double Ffp = 0.5*(Fc[i][j]+Fc[ip][j]);
                        double Ffm = 0.5*(Fc[i][j]+Fc[im][j]);
                        double Fgp = 0.5*(Fc[i][j]+Fc[i][jp]);
                        double Fgm = 0.5*(Fc[i][j]+Fc[i][jm]);
                        dfp[i][j] = ( Ffp*(phi[ip][j]-phi[i][j]) - Ffm*(phi[i][j]-phi[im][j]) )/(dx*dx)
                                  + ( Fgp*(phi[i][jp]-phi[i][j]) - Fgm*(phi[i][j]-phi[i][jm]) )/(dy*dy);
                    }
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
                    double M=(1-c[i][j])*M_A+c[i][j]*M_B;
                   // M=8;
                   // double K=(1-c[i][j])*K_A+c[i][j]*K_B;
                   // double Cp=(1-c[i][j])*Cp_A+c[i][j]*Cp_B;
                    //double L= (1-c[i][j])*L_A+c[i][j]*L_B;


                    double t1 = (eps[i][jp]*d_eps[i][jp]*phi_dx[i][jp] - eps[i][jm]*d_eps[i][jm]*phi_dx[i][jm]) / 2 / dy;
                    double t2 = (eps[ip][j]*d_eps[ip][j]*phi_dy[ip][j] - eps[im][j]*d_eps[im][j]*phi_dy[im][j]) / 2 / dx;


                    double g = phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                    double g_dash = 2 * phi_old * (1 - phi_old) * (1 - phi_old) - 2 * phi_old * phi_old * (1 - phi_old);
                    double g_ddash=2*(1 - phi_old) * (1 - phi_old)-8*phi_old * (1 - phi_old)+2*phi_old*phi_old;


                    double TmA_eff = Tm_A, TmB_eff = Tm_B;
                    if (UV_MODE == 1) {
                        double x = i * UV_DX_PHYS;
                        double dp = uv_dTm_nd * sin(uv_omega * t_now - uv_k * x);
                        TmA_eff = Tm_A + dp;
                        TmB_eff = Tm_B + dp;
                    } else if (UV_MODE == 2) {
                        double x = i * UV_DX_PHYS;
                        double dp = uv_dTm_nd * sin(uv_k * x) * sin(uv_omega * t_now);
                        TmA_eff = Tm_A + dp;
                        TmB_eff = Tm_B + dp;
                    }
                    double H_A=w_A*g_dash+30*g*L_A*(TmA_eff-T)/TmA_eff/T;
                    double H_B=w_B*g_dash+30*g*L_B*(TmB_eff-T)/TmB_eff/T;

                    double dH_A=w_A*g_ddash+30*g_dash*L_A*(TmA_eff-T)/TmA_eff/T;
                    double dH_B=w_B*g_ddash+30*g_dash*L_B*(TmB_eff-T)/TmB_eff/T;


                    double t3 =(1-c[i][j])*H_A+c[i][j]*H_B;

                    double dc_dp = 30*g*(D_lA-D_sA);


                    double random = -1 + (double)rand() / RAND_MAX * 2;
                    //double noise = 16 * alfa * random * phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                   double noise = -M*alfa*random*16*g*t3;



                    phi[i][j] = phi_old + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    //T[i][j] = T_old + dt * K/Cp * lap_T[i][j] + L / Cp * 30 * g * (phi[i][j] - phi_old);
                    //c[i][j]=c_old+dt*D*(lap_c[i][j]+vm/R*c[i][j]*(1-c[i][j])*(H_B-H_A)*lap_phi[i][j]);
                   double t4=(dc_dp+vm/R*D*(H_B-H_A)*(1-2*c_old))*(phi_dx[i][j]*c_dx[i][j]+phi_dy[i][j]*c_dy[i][j]);
                    double t5=D*c_old*(1-c_old)*vm/R*(H_B-H_A);
                    double t6=dc_dp*(H_B-H_A)+(dH_B-dH_A)*D;
                    double t7=vm/R*c_old*(1-c_old)*t6;

                   /* double t4=(dc_dp+D*(H_B-H_A)*(1-2*c_old))*(phi_dx[i][j]*c_dx[i][j]+phi_dy[i][j]*c_dy[i][j]);
                    double t5=D*c_old*(1-c_old)*(H_B-H_A);
                    double t6=dc_dp*(H_B-H_A)+(dH_B-dH_A)*D;
                    double t7=c_old*(1-c_old)*t6;*/


                    if (CONS_MODE) {
                        c[i][j]=c_old+dt*(dfc[i][j]+dfp[i][j]+t4);
                    } else {
                        c[i][j]=c_old+dt*(D*lap_c[i][j]+t5*lap_phi[i][j]+t7*lap_phi[i][j]+t4);
                    }







                }

            }
         // printf(".%d", iter);
if (iter % 100000 == 0) {
            char fn1[48], fn2[48];
            sprintf(fn1, "bnn_uv_phi_%d.csv", iter/100000);
            sprintf(fn2, "bnn_uv_c_%d.csv", iter/100000);
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
            double sc=0.0, sc0=0.0, mn=1e9, mx=-1e9;
            for (int a=0;a<Nx;a++) for (int b=0;b<Ny;b++){ sc+=c[a][b]; if(c[a][b]<mn)mn=c[a][b]; if(c[a][b]>mx)mx=c[a][b]; }
            printf("  AUDIT step=%d sum(c)=%.10e mean(c)=%.10e min=%.5f max=%.5f solid=%.4f\n",
                   iter, sc, sc/(Nx*Ny), mn, mx, (double)count_phi(phi,Nx,Ny)/(Nx*Ny));
          }

    }

for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            printf( "%e,    ", phi[i][j]);
        }

    }



char outphi[64], outc2[64];
 sprintf(outphi, "phib_uv_PA%.0e_f%.0e.csv", UV_PA, UV_FREQ);
 sprintf(outc2, "cb_uv_PA%.0e_f%.0e.csv", UV_PA, UV_FREQ);
 char filename[64]; strcpy(filename, outphi);

    // open the file for writing
    FILE *fp = fopen(filename, "w");

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





    char filename2[64]; strcpy(filename2, outc2);

    // open the file for writing
    FILE *fp2 = fopen(filename2, "w");

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
    free(Dco[i]); free(Fc[i]); free(dfc[i]); free(dfp[i]);
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
free(Dco); free(Fc); free(dfc); free(dfp);

}


