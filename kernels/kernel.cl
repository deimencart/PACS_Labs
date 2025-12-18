__kernel void gaussienFiltre(
    __global float *in,
    __global float *filtregaussien,
    __global float *out,
    const unsigned int gaussienSize,
    const unsigned int width,
    const unsigned int a)  
{
    int h = get_global_id(0);
    int x = h % width;
    int y = h / width;

    float sum = 0.0f;
    float sumWeights = 0.0f;

    for (int i = -((int)a); i <= (int)a; i++) {
        for (int j = -((int)a); j <= (int)a; j++) {
            int yy = y + i;
            int xx = x + j;
            if (yy >= 0 && xx >= 0 && yy < width && xx < width) {
                float w = filtregaussien[(i + a) * gaussienSize + (j + a)];
                sum += w * in[yy * width + xx];
                sumWeights += w;
            }
        }
    }
    sum =sum / sumWeights;
    out[h] = sum;
}