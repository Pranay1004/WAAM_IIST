
// C Code for OpenMP implementation for Pure Ni


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main() {

    // Define model parameters
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
    int Nx = 500;
    int Ny = 500;

    double dx = 2e-8;
    double dy = 2e-8;

    // Define time integration parameters
    int nstep = 1e6;
    double dt = 1e-12;

  // double phi[Nx+10][Ny+10], T[Nx+10][Ny+10];
   double** phi = (double**)malloc(Nx * sizeof(double*));
   double** T = (double**)malloc(Nx * sizeof(double*));
   for (int i = 0; i < Nx; i++) {
        phi[i] = (double*)malloc(Ny * sizeof(double));
        T[i] = (double*)malloc(Ny * sizeof(double));
    }

/////////////////////////////////////////////     Nucleation     //////////////////////////////////////////

    int r = 5; // radius of nucleus
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            double distance = sqrt((i-Nx/2) * (i-Nx/2) + (j-Ny/2) * (j-Ny/2));

            if (distance < r)
                {
                phi[i][j] = 1.0;        //Nucleus at the centre
                }
               // phi[Nx-i-1][j]=1.0;

        }
    }

    for(int j=0;j<Nx;j++)
    {
    for (int i=0;i<Ny; i++)
    {
        T[j][i]=Tm-170;              //Supercooling
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
                    double t3 = w * g_dash + 30 * g * H / Tm * (T_old - Tm);
                    phi[i][j] = phi[i][j] + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    T[i][j] = T_old + dt * D * lap_T[i][j] + H / Cp * 30 * g * (phi[i][j] - phi_old);

                }

            }
//printf(".%d", iter);



    }
    
     

    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            printf("%e    ", phi[i][j]);
        }
        printf("\n");
    }


char filename[12] = "phi.csv";

    // open the file for writing
    FILE *fp1 = fopen(filename, "w");

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

    char filename2[12] = "T.csv";

    // open the file for writing
    FILE *fp2 = fopen(filename2, "w");

    // write to the excel file
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            fprintf(fp2, "%e,    ", T[i][j]);
        }
        fprintf(fp2, "\n");
    }

    // close the file
    fclose(fp2);
   


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

