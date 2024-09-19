package Assignment1;
public class Book{
    enum Type{
        INLIB,
        BORROWED,
        ABANDONED
    };
    String name;
    Type type;
    public Book(String name)
    {
        this.name = name;
        this.type = Type.INLIB;
    }
    public boolean borrow()
    {
        if(this.type != Type.INLIB)
            return false;
        this.type = Type.BORROWED;
        return true;
    }
    public boolean ret()
    {
        this.type = Type.INLIB;
        return true;
    }
    public boolean abandon()
    {
        if(this.type == Type.ABANDONED)
            return false;
        this.type = Type.ABANDONED;
        return true;
    }
}