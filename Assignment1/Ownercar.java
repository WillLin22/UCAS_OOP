package Assignment1;

public class Ownercar {
    String name;
    int type;
    int distance;
    public Ownercar(String name, int type, int license, int distance)
    {
        this.name = name;
        this.type = type;
        this.distance = distance;
    }
    String getname()
    {
        return this.name;
    }
    int gettype()
    {
        return this.type;
    }
    int getdistance()
    {
        return this.distance;
    }
    void adddistance(int from, int to)
    {
        this.distance += to - from;
    }
}