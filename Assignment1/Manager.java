package Assignment1;
import java.util.ArrayList;
public class Manager{
    ArrayList<Book> lib;
    public Manager()
    {
        this.lib = new ArrayList<>();
    }
    public boolean addbook(String name)
    {
        this.lib.add(new Book(name));
        return true;
    }
    public boolean removebook(String name)
    {
        for(var e : lib)
        {
            if(e.name == name)
            {
                e.abandon();
                lib.remove(e);
                return true;
            }
        }
        return false;
    }
    public Book borrowbook(String name)
    {
        for(var e : lib)
        {
            if(e.name == name)
            {
                e.borrow();
                lib.remove(e);
                return e;
            }
        }   
        return null;
    }
    public boolean retbook(Book b)
    {
        b.ret();
        lib.add(b);
        return true;
    }

}