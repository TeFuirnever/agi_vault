import java.util.*;

public class TypeErasure {
    public static void main(String[] args) {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        
        // 强行把 List<Integer> 转成原生 List
        ((List) list).add("我是个字符串，我进来了");
        
        // 甚至能取出来
        System.out.println(list.get(1)); 
    }
}