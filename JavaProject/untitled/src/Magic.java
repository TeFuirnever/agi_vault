import java.lang.reflect.Field;

public class Magic {
    public static void main(String[] args) {
        try {
            //以此致敬《1984》
            Class<?> cache = Integer.class.getDeclaredClasses()[0]; 
            Field c = cache.getDeclaredField("cache");
            c.setAccessible(true);
            Integer[] array = (Integer[]) c.get(cache);
            
            // 我们要把 4 变成 5
            // Integer cache通常是从 -128 到 127
            // 数组索引：value + 128
            // 4 的索引是 132， 5 的索引是 133
            array[132] = array[133]; 

            System.out.printf("2 + 2 = %d", 2 + 2); 
        } catch (NoSuchFieldException | IllegalAccessException | IndexOutOfBoundsException e) {
            System.err.println("无法修改Integer缓存: " + e.getMessage());
            e.printStackTrace();
        } catch (java.lang.reflect.InaccessibleObjectException e) {
            System.err.println("无法访问内部API，请使用 --add-opens 参数运行: " + e.getMessage());
            System.out.println("建议运行命令: java --add-opens java.base/java.lang=ALL-UNNAMED Magic");
        } catch (Exception e) {
            System.err.println("发生未知错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}