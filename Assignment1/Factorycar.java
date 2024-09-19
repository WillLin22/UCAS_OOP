package Assignment1;

public class Factorycar{
    String name;
    int type;
    int index1, index2, index3;
    int price;
    public Factorycar(String name, int type, int index1, int index2, int index3, int price)
    {
        this.name = name;
        this.type = type;
        this.index1 = index1;
        this.index2 = index2;
        this.index3 = index3;
        this.price = price;
    }
    String getname()
    {
        return this.name;
    }
    int gettype()
    {
        return this.type;
    }
    int getindex1()
    {
        return this.index1;
    }
    int getindex2()
    {
        return this.index2;
    }
    int getindex3()
    {
        return this.index3;
    }
    int getprice()
    {
        return this.price;
    }
    void setprice(int price)
    {
        this.price = price;
    }
    void setname(String name)
    {
        this.name = name;
    }
    void settype(int type)
    {
        this.type = type;
    }
    void setindex1(int index1)
    {
        this.index1 = index1;
    }
    void setindex2(int index2)
    {
        this.index2 = index2;
    }
    void setindex3(int index3)
    {
        this.index3 = index3;
    }
}