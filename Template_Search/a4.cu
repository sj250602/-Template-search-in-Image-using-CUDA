#include <bits/stdc++.h>

using namespace std;

__device__ float bi_linear(float* d_image,int row,int col, float x, float y, int v){
    int x1 = floor(x);
    int y1 = ceil(y);
    return d_image[y1*col*3+x1*3+v]*(x1+1-x)*abs(-y1+1+y)+d_image[y1*col*3+(x1+1)*3+v]*(x-x1)*abs(y1-1-y)+d_image[(y1-1)*col*3+x1*3+v]*(x1+1-x)*abs(y-y1)+d_image[(y1-1)*col*3+(x1+1)*3+v]*(x-x1)*abs(y-y1);
}

__device__ float rotation_x(float x, float y, int angle){
    if(angle==45){
        return (x-y)/sqrt(2.0);
    }else{
        return (x+y)/sqrt(2.0);
    }
}

__device__ float rotation_y(float x, float y, int angle){
    if(angle==45){
        return (x+y)/sqrt(2.0);
    }else{
        return (y-x)/sqrt(2.0);
    }
}

__device__ float rmsd(int bit,int q_row,int q_col,float*d_image,float*q_image,int r,int c,int d_row,int d_col){
    if(bit==0){
        float res =0;
        for(int i=r;i>r-q_row;i--){
            for(int j=c;j<c+q_col;j++){
                for(int k=0;k<3;k++){
                    res+=pow(d_image[3*d_col*(i)+3*(j)+k]-q_image[3*q_col*(q_row-(r-i)-1)+ 3*(j-c)+k],2);
                }
            }
        }

        res/=(q_row*q_col*3);
        return sqrt(res);
    }
    else if(bit==1){
        float res =0;
        for(int i=r;i>r-q_row;i--){
            for(int j=c;j<c+q_col;j++){
                float x = c+rotation_x(j-c,r-i,45);
                float y = r-rotation_y(j-c,r-i,45);
                for(int k=0;k<3;k++){
                    float red = bi_linear(d_image,d_row,d_col,x,y,k);
                    res+=pow(red-q_image[3*q_col*(q_row-1-(r-i))+ 3*(j-c) +k],2);
                }
            }
        }
        res/=(q_row*q_col*3);
        return sqrt(res);
    }else if(bit==2){
        float res =0;
        for(int i=r;i>r-q_row;i--){
            for(int j=c;j<c+q_col;j++){
                float x = c+rotation_x(j-c,r-i,-45);
                float y = r-rotation_y(j-c,r-i,-45);
                for(int k=0;k<3;k++){
                    float red = bi_linear(d_image,d_row,d_col,x,y,k);
                    res+=pow(red-q_image[3*q_col*(q_row-1-(r-i))+ 3*(j-c) +k],2);
                }
            }
        }
        res/=(q_row*q_col*3);
        return sqrt(res);
    }
    return 0;
}

__global__ void start_comp(float* d_image,float* q_image,int d_row,int d_col,int q_row,int q_col,float q_avg,float th2,float th1,float* res){

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int pixel = 3*(bid*128+tid);
    int d_row_num = pixel/(3*d_col);
    int d_col_num = (pixel/3)%(d_col);
    if(pixel<d_row*d_col*3){
        //for zero
        if(d_row_num-q_row>=-1 && d_col_num+q_col<=d_col){
            float d_avg=0.0;
            for(int i=d_row_num;i>d_row_num-q_row;i--){
                for(int j=d_col_num;j<(d_col_num+q_col);j++){
                    d_avg+=((d_image[d_col*i*3+j*3]+d_image[d_col*i*3+j*3+1]+d_image[d_col*i*3+j*3+2])/3);
                }
            }
            d_avg/=(q_row*q_col);
            if(abs(d_avg-q_avg)<th2){
                
                float d_rmsd0 = rmsd(0,q_row,q_col,d_image,q_image,d_row_num,d_col_num,d_row,d_col);
                if(d_rmsd0<th1){
                    res[3*d_row_num*d_col+d_col_num*3] = d_rmsd0;
                }
            }
        }

        //for 45 
        float B_x = d_col_num+rotation_x(q_col-1,0,45);
        float B_y = d_row-1-d_row_num+rotation_y(q_col-1,0,45);


        float C_x = d_col_num+rotation_x(q_col-1,q_row-1,45);
        float C_y = d_row-1-d_row_num+rotation_y(q_col-1,q_row-1,45);


        float D_x = d_col_num+rotation_x(0,q_row-1,45);
        float D_y = d_row-1-d_row_num+rotation_y(0,q_row-1,45);

        int x1 = floor(D_x);
        int x2 = ceil(B_x);
        int y2 = ceil(C_y);
        
        if(x1>=0&& x2<d_col&&y2-d_row+1<=0&&d_row_num>=0){
            float d_avg = 0.0;
            for(int r=d_row_num;r>=d_row-1-y2;r--){
                for(int c=x1;c<=x2;c++){
                    d_avg+=((d_image[d_col*r*3+c*3]+d_image[d_col*r*3+c*3+  1]+d_image[d_col*r*3+c*3+2])/3);
                }
            }
            d_avg/=abs(d_row_num-(d_row-1-y2)+1)*abs(x2-x1+1);

            if(abs(d_avg-q_avg)<th2){
                float d_rmsd1 = rmsd(1,q_row,q_col,d_image,q_image,d_row_num,d_col_num,d_row,d_col);
                if(d_rmsd1<th1){
                    
                    res[3*d_row_num*d_col+d_col_num*3+1] = d_rmsd1;
                    
                }
            }
        }
        // //-45

        float b_x = d_col_num+rotation_x(q_col-1,0,-45);
        float b_y = d_row-1-d_row_num+rotation_y(q_col-1,0,-45);


        float c_x = d_col_num+rotation_x(q_col-1,q_row-1,-45);
        float c_y = d_row-1-d_row_num+rotation_y(q_col-1,q_row-1,-45);


        float d_x = d_col_num+rotation_x(0,q_row-1,-45);
        float d_y = d_row-1-d_row_num+rotation_y(0,q_row-1,-45);

        int y11 = ceil(d_y);
        int x22 = ceil(c_x);
        int y22 = floor(b_y);

        if(y22>=0&& x22<d_col&&y11-d_row+1<=0&&d_col_num>=0){
            float d_avg = 0.0;
            for(int r=d_row-1-y22;r>=d_row-1-y11;r--){
                for(int c=d_col_num;c<=x22;c++){
                    d_avg+=((d_image[d_col*r*3+c*3]+d_image[d_col*r*3+c*3+  1]+d_image[d_col*r*3+c*3+2])/3);
                }
            }
            d_avg/=abs(y11-y22+1)*abs(x22-d_col_num+1);
            float abs_d_avg = abs(d_avg-q_avg);

            if(abs_d_avg<th2){
                float d_rmsd2 = rmsd(2,q_row,q_col,d_image,q_image,d_row_num,d_col_num,d_row,d_col);

                if(d_rmsd2<th1){
                    
                    res[3*d_row_num*d_col+d_col_num*3+2] = d_rmsd2;
                    
                }
            }
        }
    }
}

class ans_tuple{
    public:
        int row,col,bit;
        float dis;
    ans_tuple(){
    }

    ans_tuple(float a,int b,int c,int d){
        dis = a;
        row = b;
        col = c;
        bit = d;
    }
};

struct grtr_tuple{
    bool operator()(const ans_tuple &t1,const ans_tuple &t2){
        return t1.dis<t2.dis;
    }
};

int main(int argc, char* argv[])
{
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);cout.tie(NULL);
  assert (argc > 5);
  string d_str = argv[1];
  string q_str = argv[2];
  float th1 = stof(argv[3]);
  float th2 = stof(argv[4]);
  int n = stoi(argv[5]);

  int d_row, d_col, q_row, q_col,k=0;
  float *d_image,*q_image;
  float q_avg;
  ifstream d_file(d_str),q_file(q_str);
  string str;
  getline(d_file,str);
  stringstream ss(str);
  string word;
  ss>>word;
  d_row = stoi(word);
  ss>>word;
  d_col = stoi(word);
  d_image = new float[d_row*d_col*3];


  getline(d_file,str);
  d_file.close();
  stringstream s(str);
  for(int i=0;i<d_row;i++){
      for(int j=0;j<d_col;j++){
          float a,b,c;
          s>>word;
          a = stof(word);
          s>>word;
          b = stof(word);
          s>>word;
          c = stof(word);
          d_image[k++] = a;
          d_image[k++] = b;
          d_image[k++] = c;
      }
  }

  k=0;
  getline(q_file,str);
  stringstream ss1(str);
  ss1>>word;
  q_row = stoi(word);
  ss1>>word;
  q_col = stoi(word);
  q_image = new float[q_row*q_col*3];
  getline(q_file,str);
  q_file.close();
  stringstream s1(str);
  for(int i=0;i<q_row;i++){
      vector<vector<float>> vect;
      vector<float> grey;
      for(int j=0;j<q_col;j++){
          float a,b,c;
          s1>>word;
          a = stof(word);
          s1>>word;
          b = stof(word);
          s1>>word;
          c = stof(word);
          q_image[k++] = a;
          q_image[k++] = b;
          q_image[k++] = c;
          q_avg += ((a+b+c)/3);
      }
  }
    
  q_avg/=(q_row*q_col);
  
  int d_isc = 3*d_row*d_col;
  int d_gsc = d_row*d_col;
  int q_isc = 3*q_row*q_col;
  float* o_d_image,*o_q_image,*res;
  float* res_c = new float[d_isc];

  cudaMalloc(&o_d_image,d_isc*sizeof(float));
  cudaMalloc(&o_q_image,q_isc*sizeof(float));
  cudaMalloc(&res,d_isc*sizeof(float));

  for(int i=0;i<d_isc;i++){
      res_c[i] = -100000.0;
  }
  
  cudaMemcpy(o_d_image,d_image,d_isc*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(o_q_image,q_image,q_isc*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(res,res_c,d_isc*sizeof(float),cudaMemcpyHostToDevice);  

  start_comp<<<(d_gsc+127)/128,128>>>(o_d_image,o_q_image,d_row,d_col,q_row,q_col,q_avg,th2,th1,res);
  cudaMemcpy(res_c,res,d_isc*sizeof(float),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  priority_queue <ans_tuple,vector<ans_tuple>,grtr_tuple> pq;
  for(int i=0;i<d_row;i++){
    for(int j=0;j<d_col;j++){
        for(int k=0;k<3;k++){
            float rmsd_error = res_c[3*i*d_col+3*j+k];
            if(rmsd_error!=-100000){
                pq.push(ans_tuple(rmsd_error,d_row-1-i,j,k));
                if(pq.size()>n){
                    pq.pop();
                }
            }
        }
    }
  }

  vector<ans_tuple> ans(n);
  int len = n-1;
  while(!pq.empty()){
      ans[len--] = pq.top();
      if(ans[len+1].bit==1){
        ans[len+1].bit = 45;
      }else if(ans[len+1].bit==2){
        ans[len+1].bit = -45;
      }
      pq.pop();
  }

  cudaFree(o_d_image);
  cudaFree(o_q_image);
  cudaFree(res);
  free(d_image);
  free(q_image);

  ofstream output("output.txt");
  for(int i=0;i<n;i++){
    output<<ans[i].row<<" "<<ans[i].col<<" "<<ans[i].bit<<"\n";
  }
  output.close();


}
