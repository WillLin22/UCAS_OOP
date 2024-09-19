package Assignment1;
import java.util.ArrayList;

public class Student{
    ArrayList<Book> borrowlist;
    String name;
    Manager man;
    public Student(String s, Manager m)
    {
        this.name = s;
        this.borrowlist = new ArrayList<>();
        this.man = m;
    }
    public boolean borrow(String bookname)
    {
        Book b = man.borrowbook(bookname);
        if(b == null) return false;
        borrowlist.add(b);
        return true;
    }
    public boolean ret(Book b)
    {
        for(var e : borrowlist)
        {
            if(e == b)
            {
                e.ret();
                borrowlist.remove(e);
                man.retbook(e);
                return true;
            }
        }
        return false;
    }
}