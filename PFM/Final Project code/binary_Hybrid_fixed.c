
// This is code for Hybrid (OpenMP+MPICH) implementation for pure Ni.
// FIXED: Contiguous memory allocation for MPI Send/Recv of 2D arrays.



#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

// Helper: allocate Nx*Ny contiguous doubles, return row pointers
static double** alloc_contiguous(int Nx, int Ny) {
    double* data = (double*)malloc((size_t)Nx * Ny * sizeof(double));
    double** rows = (double**)malloc(Nx * sizeof(double*));
    for (int i = 0; i < Nx; i++) {
        rows[i] = data + (size_t)i * Ny;
    }
    memset(data, 0, (size_t)Nx * Ny * sizeof(double));
    return rows;
}

static void free_contiguous(double** rows) {
    if (rows) {
        free(rows[0]); // free the single contiguous block
        free(rows);    // free the pointer array
    }
}

int main(int argc, char* argv[]) {



    // Define model parameters
    double sigma0 = 0.37;
    double nu = 2;
    double Tm = 1728;
    double H = 2.35e9;
    double D = 1.55e-5;
    double Cp = 5.42e6;
    double lambda = 4e-8;

    int aniso = 4;
    double theta0 = M_PI / 4.0;

    // Define other parameters
    double w = 6.6 * sigma0 / lambda;
    double eps0 = sqrt(2.73 * lambda * sigma0);
    double M = Tm * nu / 2.73 / H / lambda;
    double delta = 0.025;
    double alfa = 0.025;

    // Define grid parameters
    int Nx = 1000;
    int Ny = 1000;

    double dx = 2e-8;
    double dy = 2e-8;

    // Define time integration parameters
    int nstep = 1e3;
    double dt = 1e-12;

    double** phi = alloc_contiguous(Nx, Ny);
    double** T = alloc_contiguous(Nx, Ny);

    int r = 5; // radius of nucleus
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            double distance = sqrt((i) * (i) + (j) * (j));

            if (distance < r)
                {
                phi[i][j] = 1.0;
                }

            T[i][j] = 1448.0;
        }
    }

    double** lap_phi = alloc_contiguous(Nx, Ny);
    double** lap_T = alloc_contiguous(Nx, Ny);
    double** phi_dx = alloc_contiguous(Nx, Ny);
    double** phi_dy = alloc_contiguous(Nx, Ny);
    double** eps = alloc_contiguous(Nx, Ny);
    double** d_eps = alloc_contiguous(Nx, Ny);

    double** phi1 = alloc_contiguous(Nx, Ny);
    double** T1 = alloc_contiguous(Nx, Ny);
    double** lap_phi1 = alloc_contiguous(Nx, Ny);
    double** lap_T1 = alloc_contiguous(Nx, Ny);
    double** phi_dx1 = alloc_contiguous(Nx, Ny);
    double** phi_dy1 = alloc_contiguous(Nx, Ny);
    double** eps1 = alloc_contiguous(Nx, Ny);
    double** d_eps1 = alloc_contiguous(Nx, Ny);


    int n = 0;  //  Step Counter

clock_t start, end;
    /* Store start time here */
    start = clock();

    int rank, n_ranks, numbers_per_rank,nz;
 int my_first, my_last;
 int numbers = Nx;
 int a,b;

    // First call MPI_Init
 MPI_Init(&argc, &argv);
 // Get my rank and the number of ranks
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);        //get my process id
 MPI_Comm_size(MPI_COMM_WORLD,&n_ranks);      //get number of processes
 MPI_Status status;

 // Calculate the number of iterations for each rank
 numbers_per_rank = floor(numbers/n_ranks);

 // Figure out the first and the last iteration for this rank
 my_first = rank * numbers_per_rank;
 my_last = my_first + numbers_per_rank-1;


    for (int iter = 1; iter <= nstep; iter++)
{

            if (rank==0)
            {
                for(int i=1; i<n_ranks;i++)
                {

                my_first = i * numbers_per_rank;
                my_last = my_first + numbers_per_rank-1;
                MPI_Send(&my_first, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&my_last, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(phi[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(T[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(lap_phi[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(lap_T[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(phi_dx[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(phi_dy[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(eps[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(d_eps[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);



                }

                for(int i=1; i<n_ranks;i++)
                {
                MPI_Recv(&a, 1, MPI_INT, i,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&b, 1, MPI_INT, i,0, MPI_COMM_WORLD,&status);
                MPI_Recv(&nz, 1, MPI_INT, i,0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(T1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_phi1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_T1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi_dx1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi_dy1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(eps1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(d_eps1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);

                      for(int j=a;j<=b;j++)
                      {
                            for(int k=0;k<Ny;k++)

                            {

                                lap_phi[j][k]=lap_phi1[j-a][k];
                                lap_T[j][k]=lap_T1[j-a][k];
                                phi_dx[j][k]=phi_dx1[j-a][k];
                                phi_dy[j][k]=phi_dy1[j-a][k];
                                eps[j][k]=eps1[j-a][k];
                                d_eps[j][k]=d_eps1[j-a][k];
                             }


                      }



                }

               for (int i = 0; i < Nx; i++)
                {
                    for (int j = 0; j < Ny; j++)
                     {




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

                    double phi_new = phi_old + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    double T_new = T_old + dt * D * lap_T[i][j] + H / Cp * 30 * g * (phi[i][j] - phi_old);

                    phi[i][j]=phi_new;
                    T[i][j]=T_new;


                    }
               }

            }
            else
            {
                MPI_Recv(&my_first, 1, MPI_INT,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&my_last, 1, MPI_INT,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(T[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_phi[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_T[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi_dx[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi_dy[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(eps[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(d_eps[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);

                for(int i = my_first; i < my_last; i++)
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

                                    lap_phi[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                                    p1 = T[ip][j]; // right
                                    p2 = T[im][j]; // Left
                                    p3 = T[i][jp]; // up
                                    p4 = T[i][jm]; // down
                                    p5 = T[i][j];  // centre

                                    lap_T[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                                    // Calculating gradient

                                    phi_dx[i-my_first][j] = (phi[ip][j] - phi[im][j]) / (2 * dx);
                                    phi_dy[i-my_first][j] = (phi[i][jp] - phi[i][jm]) / (2 * dy);

                                    // Theta

                                    double theta = atan2(phi_dy[i][j], phi_dx[i][j]);

                                    // epsilon

                                    eps[i-my_first][j] = eps0 * (1 + delta * cos(aniso * (theta - theta0)));

                                    // deps/dtheta

                                    d_eps[i-my_first][j] = -eps0 * delta * aniso * sin(aniso * (theta - theta0));

                            }
                  }

                 nz=my_last-my_first+1;

                MPI_Send(&my_first, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&my_last, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&nz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(phi[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(T[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(lap_phi[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(lap_T[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(phi_dx[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(phi_dy[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(eps[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(d_eps[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);



           }


            // Time Evolution


}
//printf(".%d", iter);


                if(rank==0)
              {

                         end = clock();
                            /* Get the time taken by program to execute in seconds */
                            double duration = ((double)end - start)/CLOCKS_PER_SEC;

                            printf("Time taken to execute in seconds : %f", duration);


                        char filename[10] = "phi.csv";

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

                 }




                free_contiguous(phi);
                free_contiguous(T);
                free_contiguous(lap_phi);
                free_contiguous(lap_T);
                free_contiguous(phi_dx);
                free_contiguous(phi_dy);
                free_contiguous(eps);
                free_contiguous(d_eps);
                free_contiguous(phi1);
                free_contiguous(T1);
                free_contiguous(lap_phi1);
                free_contiguous(lap_T1);
                free_contiguous(phi_dx1);
                free_contiguous(phi_dy1);
                free_contiguous(eps1);
                free_contiguous(d_eps1);

 MPI_Finalize();
 return 0;
}
