package org.example.controller;

import org.springframework.web.bind.annotation.*;
import java.util.*;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @GetMapping
    public List<Map<String, Object>> getUsers() {
        List<Map<String, Object>> users = new ArrayList<>();
        users.add(Map.of("id", 1, "name", "Alice"));
        users.add(Map.of("id", 2, "name", "Bob"));
        return users;
    }
}
