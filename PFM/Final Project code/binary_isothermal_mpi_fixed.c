#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

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
        free(rows[0]);
        free(rows);
    }
}

// Binary alloy Ni and Cu
int main(int argc, char* argv[]) {

    // Physical data (Ni)

    double sigma_A = 0.37;
    double nu_A = 2e2;
    double Tm_A = 1728;
    double L_A = 2.35e9;
    double Cp_A = 5.42e6;
    double D_lA=1e-9;
    double D_sA=1e-13;


    // Physical data (Cu)

    double sigma_B = 0.29;
    double nu_B = 2.47e2;
    double Tm_B = 1358;
    double L_B = 1.728e9;
    double Cp_B = 3.96e6;
    double D_lB=1e-9;
    double D_sB=1e-13;



    int aniso = 4;
    double theta0 = 0;
    double lambda_A=4.9e-6/2;
    double lambda_B=4.9e-6/2;

    double w_A = 6.6 * sigma_A / lambda_A;
    double eps0_A = sqrt(2.73 * lambda_A * sigma_A);
    double M_A = Tm_A * nu_A / 2.73 / L_A / lambda_A;

    double w_B = 6.6 * sigma_B/ lambda_B;
    double eps0_B = sqrt(2.73 * lambda_B * sigma_B);
    double M_B = Tm_B * nu_B / 2.73 / L_B / lambda_B;

    double delta = 0.04;
    double alfa = 0.4;
    double vm=7.42e-6;
    double R=8.314;

    int Nx = 512;
    int Ny = 512;

    double dx = lambda_A/4;
    double dy = dx;

    double nstep = 1e6;
    double dt = dx*dx*.1;

    double** phi = alloc_contiguous(Nx, Ny);
    double** T = alloc_contiguous(Nx, Ny);
    double** c = alloc_contiguous(Nx, Ny);

    int r = 10;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            double distance = sqrt((i-Nx/2) * (i-Nx/2) + (j-Ny/2) * (j-Ny/2));

            if (distance < r)
                {
                phi[i][j] = 1.0;
                }

            T[i][j] = 1554.0;
            c[i][j]=.4083;
        }
    }

    double** lap_phi = alloc_contiguous(Nx, Ny);
    double** lap_c = alloc_contiguous(Nx, Ny);
    double** phi_dx = alloc_contiguous(Nx, Ny);
    double** phi_dy = alloc_contiguous(Nx, Ny);
    double** eps = alloc_contiguous(Nx, Ny);
    double** d_eps = alloc_contiguous(Nx, Ny);

    double** phi1 = alloc_contiguous(Nx, Ny);
    double** T1 = alloc_contiguous(Nx, Ny);
    double** c1 = alloc_contiguous(Nx, Ny);
    double** lap_phi1 = alloc_contiguous(Nx, Ny);
    double** lap_c1 = alloc_contiguous(Nx, Ny);
    double** phi_dx1 = alloc_contiguous(Nx, Ny);
    double** phi_dy1 = alloc_contiguous(Nx, Ny);
    double** eps1 = alloc_contiguous(Nx, Ny);
    double** d_eps1 = alloc_contiguous(Nx, Ny);

    int n = 0;

printf("Started");

clock_t start, end;
    start = clock();

    int rank, n_ranks, numbers_per_rank,nz;
 int my_first, my_last;
 int numbers = Nx;
 int a,b;

 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 MPI_Comm_size(MPI_COMM_WORLD,&n_ranks);
 MPI_Status status;

 numbers_per_rank = floor(numbers/n_ranks);

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
                MPI_Send(c[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(lap_phi[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(lap_c[0], Nx*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
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
                MPI_Recv(c1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_phi1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_c1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi_dx1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi_dy1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(eps1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(d_eps1[0], nz*Ny, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&status);

                      for(int j=a;j<=b;j++)
                      {
                            for(int k=0;k<Ny;k++)

                            {

                                lap_phi[j][k]=lap_phi1[j-a][k];
                                 lap_c[j][k]=lap_c1[j-a][k];
                                phi_dx[j][k]=phi_dx1[j-a][k];
                                phi_dy[j][k]=phi_dy1[j-a][k];
                                eps[j][k]=eps1[j-a][k];
                                d_eps[j][k]=d_eps1[j-a][k];
                             }


                      }

                }

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

                    double phi_old = phi[i][j];
                   double T_old = T[i][j];
                   double c_old = c[i][j];


                    double p=phi_old*phi_old*phi_old*(6*phi_old*phi_old+15*phi_old-10);

                    double D=D_sA+p*(D_lA-D_sA);
                    double M=(1-c[i][j])*M_A+c[i][j]*M_B;
                    double L= (1-c[i][j])*L_A+c[i][j]*L_B;


                    double t1 = (eps[i][jp]*d_eps[i][jp]*phi_dx[i][jp] - eps[i][jm]*d_eps[i][jm]*phi_dx[i][jm]) / 2 / dy;
                    double t2 = (eps[ip][j]*d_eps[ip][j]*phi_dy[ip][j] - eps[im][j]*d_eps[im][j]*phi_dy[im][j]) / 2 / dx;
                    double g = phi_old * phi_old * (1 - phi_old) * (1 - phi_old);
                    double g_dash = 2 * phi_old * (1 - phi_old) * (1 - phi_old) - 2 * phi_old * phi_old * (1 - phi_old);

                    double H_A=w_A*g_dash+30*g*L_A*(Tm_A-T[i][j])/Tm_A;
                    double H_B=w_B*g_dash+30*g*L_B*(Tm_B-T[i][j])/Tm_B;
                    double t3 =(1-c[i][j])*H_A+c[i][j]*H_B;

                    double random = -1 + (double)rand() / RAND_MAX * 2;
                    double noise = -M*alfa*random*16*g*t3;


                    phi[i][j] = phi_old + (dt * M) * (t1 - t2 + eps[i][j] * eps[i][j] * lap_phi[i][j] - t3 + noise);
                    c[i][j]=c_old+dt*D*(lap_c[i][j]+vm/R*c[i][j]*(1-c[i][j])*(H_B-H_A)*lap_phi[i][j]);

                }

            }

            }
            else
            {
                MPI_Recv(&my_first, 1, MPI_INT,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(&my_last, 1, MPI_INT,0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(phi[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(T[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(c[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_phi[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
                MPI_Recv(lap_c[0], Nx*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&status);
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

                    double p1 = phi[ip][j];
                    double p2 = phi[im][j];
                    double p3 = phi[i][jp];
                    double p4 = phi[i][jm];
                    double p5 = phi[i][j];

                    lap_phi[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    p1 = T[ip][j];
                    p2 = T[im][j];
                    p3 = T[i][jp];
                    p4 = T[i][jm];
                    p5 = T[i][j];

                    lap_c[i-my_first][j] = (p1 + p2 + p3 + p4 - 4*p5) / (dx * dy);

                    phi_dx[i-my_first][j] = (phi[ip][j] - phi[im][j]) / (2 * dx);
                    phi_dy[i-my_first][j] = (phi[i][jp] - phi[i][jm]) / (2 * dy);

                    double theta = atan2(phi_dy[i][j], phi_dx[i][j]);

                    eps[i-my_first][j] = eps0_A * (1 + delta * cos(aniso * (theta - theta0)));

                    d_eps[i-my_first][j] = -eps0_A * delta * aniso * sin(aniso * (theta - theta0));
                            }
                  }

                 nz=my_last-my_first+1;

                MPI_Send(&my_first, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&my_last, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&nz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(phi[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(T[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(lap_phi[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(phi_dx[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(phi_dy[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(eps[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(d_eps[0], nz*Ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

           }

}


                if(rank==0)
              {

                         end = clock();
                            double duration = ((double)end - start)/CLOCKS_PER_SEC;

                            printf("Time taken to execute in seconds : %f", duration);

                        char filename[10] = "phi.csv";

                            FILE *fp = fopen(filename, "w");

                            for (int i = 0; i < Nx; i++)
                            {
                                for (int j = 0; j < Ny; j++)
                                {
                                    fprintf(fp, "%e,    ", phi[i][j]);
                                }
                                fprintf(fp, "\n");
                            }

                            fclose(fp);

                 }

                free_contiguous(phi);
                free_contiguous(T);
                free_contiguous(c);
                free_contiguous(lap_phi);
                free_contiguous(lap_c);
                free_contiguous(phi_dx);
                free_contiguous(phi_dy);
                free_contiguous(eps);
                free_contiguous(d_eps);
                free_contiguous(phi1);
                free_contiguous(T1);
                free_contiguous(c1);
                free_contiguous(lap_phi1);
                free_contiguous(lap_c1);
                free_contiguous(phi_dx1);
                free_contiguous(phi_dy1);
                free_contiguous(eps1);
                free_contiguous(d_eps1);

 MPI_Finalize();
 return 0;
}
